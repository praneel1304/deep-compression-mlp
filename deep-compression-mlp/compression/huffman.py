# huffman.py
import heapq
from collections import Counter
import numpy as np
import torch
import torch.nn as nn


# a node in our huffman tree — holds a symbol and how often it shows up
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq   = freq
        self.left   = None
        self.right  = None

    # heapq needs this to know which node is "smaller"
    def __lt__(self, other):
        return self.freq < other.freq


# build the tree bottom-up — rarest symbols end up deepest
def build_huffman_tree(freq_map):
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_map.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        # grab the two least frequent nodes and merge them
        left  = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(symbol=None, freq=left.freq + right.freq)
        parent.left  = left
        parent.right = right
        heapq.heappush(heap, parent)

    return heap[0]  # the root


# walk the tree and assign bit codes — left = 0, right = 1
def build_codebook(root):
    codebook = {}

    def _walk(node, bits):
        if node is None:
            return
        if node.symbol is not None:  # leaf node, we're done
            codebook[node.symbol] = bits or '0'
            return
        _walk(node.left,  bits + '0')
        _walk(node.right, bits + '1')

    _walk(root, '')
    return codebook


# just look up each symbol and stitch the bits together
def encode(symbols, codebook):
    return ''.join(codebook[s] for s in symbols)


# follow the tree bit by bit until we hit a leaf, then repeat
def decode(bitstring, root):
    out, node = [], root
    for bit in bitstring:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            out.append(node.symbol)
            node = root  # back to top
    return out


# how much are we actually saving?
def stats(freq_map, codebook):
    total = sum(freq_map.values())
    avg_bits   = sum(freq_map[s] * len(codebook[s]) for s in freq_map) / total
    fixed_bits = np.ceil(np.log2(max(len(freq_map), 2)))
    return {
        'avg_bits'  : round(avg_bits, 4),
        'fixed_bits': fixed_bits,
        'ratio'     : round(fixed_bits / avg_bits, 4),
        'symbols'   : total,
        'alphabet'  : len(freq_map),
    }


# run huffman on every weight layer — call this after quantization
def huffman_encode_model(model, verbose=True):
    results = {}

    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue

        # flatten + round to int (quantization should've done this already)
        symbols = param.data.cpu().numpy().flatten().round().astype(int).tolist()

        freq_map = dict(Counter(symbols))
        root     = build_huffman_tree(freq_map)
        codebook = build_codebook(root)
        encoded  = encode(symbols, codebook)
        s        = stats(freq_map, codebook)

        results[name] = {
            'codebook': codebook,
            'encoded' : encoded,
            'root'    : root,
            'stats'   : s,
        }

        if verbose:
            print(f"{name}: {s['symbols']:,} symbols | "
                  f"{s['fixed_bits']:.0f}b fixed → {s['avg_bits']}b avg | "
                  f"{s['ratio']}× smaller")

    return results


# decode everything back — good for checking nothing got mangled
def huffman_decode_model(results):
    return {
        name: decode(data['encoded'], data['root'])
        for name, data in results.items()
    }


# quick sanity check
if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(784, 300), nn.ReLU(),
        nn.Linear(300, 100), nn.ReLU(),
        nn.Linear(100, 10),
    )

    # fake quantization — snap weights to 32 centroids like k-means would
    for p in model.parameters():
        centroids = torch.linspace(p.data.min(), p.data.max(), 32)
        p.data    = centroids[((p.data.unsqueeze(-1) - centroids).abs().argmin(-1))]

    print("=== huffman encoding ===")
    results = huffman_encode_model(model)

    # make sure encode → decode gives us back the same thing
    for name, syms in huffman_decode_model(results).items():
        assert encode(syms, results[name]['codebook']) == results[name]['encoded']
        print(f"{name}: round-trip ok ✓")
