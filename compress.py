"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple
from utils import *
from huffman import HuffmanTree


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> d = build_frequency_dict(bytes([]))
    >>> d
    {}
    """
    dic = {}
    for val in text:
        dic[val] = dic.get(val, 0) + 1
    return dic


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> freq == {2: 6, 3: 4, 7: 5}
    True
    >>> build_huffman_tree({}) == HuffmanTree(None)
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    new = list(freq_dict.items())
    new.extend([((new[0][0] + 1) % 256, 1)] if len(new) == 1 else [])
    while len(new) > 1:
        lt = new.pop(new.index(min(new, key=lambda f: f[1])))
        rt = new.pop(new.index(min(new, key=lambda f: f[1])))
        if lt[1] > rt[1]:
            lt, rt = rt, lt
        lv, rv = isinstance(lt[0], HuffmanTree), isinstance(rt[0], HuffmanTree)
        new.append(
            (HuffmanTree(None, lt[0] if lv else HuffmanTree(lt[0]),
                         rt[0] if rv else HuffmanTree(rt[0])), lt[1] + rt[1]))
    return new[0][0] if new else HuffmanTree(None)


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree2 = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> d = get_codes(tree2)
    >>> d == {2: '0', 3: '10', 7: '11'}
    True
    >>> freq = {'t': 3, 3: 7, 'd': 9, 'j': 15, 'a': 16}
    >>> d = get_codes(build_huffman_tree(freq))
    >>> d == {'t': '010', 3: '011', 'd': '00', 'j': '10', 'a': '11'}
    True
    >>> get_codes(HuffmanTree(3)) == get_codes(HuffmanTree(None))
    True
    """
    if tree.is_leaf():
        return {}
    l, r = tree.left, tree.right
    l_leaf, r_leaf = l.is_leaf(), r.is_leaf()
    return {l.symbol: '0', r.symbol: '1'} if l_leaf and r_leaf else \
        {**({r.symbol: '1'} if r_leaf and not l_leaf else
            {k: '1' + v for k, v in get_codes(r).items()}),
         **({l.symbol: '0'} if l_leaf and not r_leaf else
            {k: '0' + v for k, v in get_codes(l).items()})}


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> tree.right.right.number is tree.left.left.number is None
    True
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> new = HuffmanTree(None, HuffmanTree(2), tree)
    >>> number_nodes(new)
    >>> new.number
    3
    >>> new.left.number is None and new.right.number == 2
    True
    >>> new.right.right.number == new.right.left.number + 1 == 1
    True
    >>> new.right.right.right.number is new.right.left.right.number is None
    True
    """
    if not tree.is_leaf():
        _number_helper(tree, 0)


def _number_helper(tree: HuffmanTree, num: int) -> int:
    """ Mutate <tree> internal number nodes according to <num> and
    return the next consecutive number.
    """
    if not tree.left.is_leaf():
        num = _number_helper(tree.left, num)
    if not tree.right.is_leaf():
        num = _number_helper(tree.right, num)
    tree.number = num
    return tree.number + 1


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    val = [(freq_dict.get(symbol, 0), len(code)) for
           symbol, code in get_codes(tree).items()] + [(0, 0)]
    return sum(w[0] * w[1] for w in val) / max(1, sum(f[0] for f in val))


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    val = ''.join([codes.get(byte, '') for byte in text])
    return bytes([bits_to_byte(bit) for bit in
                  [val[i: i + 8] for i in range(0, len(val), 8)]])


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    return bytes(_bytes_helper(tree)) if not tree.is_leaf() else bytes([])


def _bytes_helper(tree: HuffmanTree) -> List[int]:
    """ Return the list representation of the Huffman tree <tree> based on
    the tree's internal nodes and symbols.
    """
    ll, rl = tree.left.is_leaf(), tree.right.is_leaf()
    return (_bytes_helper(tree.left) if not ll else []) + \
           (_bytes_helper(tree.right) if not rl else []) + \
           [0 if ll else 1, tree.left.symbol if ll else tree.left.number,
            0 if rl else 1, tree.right.symbol if rl else tree.right.number]


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(1, 1, 1, 2), ReadNode(0, 5, 0, 7), ReadNode(0, 3, 0, 4)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), HuffmanTree(None, HuffmanTree(3, None, None), \
HuffmanTree(4, None, None)))
    """
    root, tree = node_lst[root_index], HuffmanTree(None)
    tree.left = generate_tree_general(node_lst, root.l_data) \
        if root.l_type else HuffmanTree(root.l_data)
    tree.right = generate_tree_general(node_lst, root.r_data) \
        if root.r_type else HuffmanTree(root.r_data)
    return tree


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> tree = build_huffman_tree(build_frequency_dict(b"worst funtion"))
    >>> number_nodes(tree)
    >>> new = bytes_to_nodes(tree_to_bytes(tree))
    >>> generate_tree_postorder(new, -1) == tree
    True
    >>> lst2 = [ReadNode(0, 1, 0, 2)]
    >>> tree2 = HuffmanTree(None, HuffmanTree(1), HuffmanTree(2))
    >>> generate_tree_postorder(lst2, 0) == tree2
    True
    >>> lst2.append(ReadNode(0, 3, 1, 1000))
    >>> tree3 = HuffmanTree(None, HuffmanTree(3), tree2)
    >>> generate_tree_postorder(lst2, 1) == tree3
    True
    """
    new, _ = node_lst.copy(), root_index
    return _postorder_helper(new)


def _postorder_helper(lst: List[ReadNode]) -> HuffmanTree:
    """ Return the HuffmanTree based on the post-ordered list <lst>.
    """
    root, tree = lst.pop(), HuffmanTree(None)
    tree.right = _postorder_helper(lst) if \
        root.r_type else HuffmanTree(root.r_data)
    tree.left = _postorder_helper(lst) if \
        root.l_type else HuffmanTree(root.l_data)
    return tree


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    codes = {v: k for k, v in get_codes(tree).items()}
    val = ''.join([byte_to_bits(bit) for bit in text])
    x = len(min(codes, key=len))
    i1, i2, lst = x * 0, x * 1, []
    while len(lst) < size:
        code = val[i1: i2]
        if code in codes:
            lst.append(codes[code])
            i1, i2 = i2, i2 + x
        else:
            i2 += 1
    return bytes(lst)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {101: 15, 97: 26, 98: 23, 99: 20, 100: 16, 69:32, 0: 100}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> left2 = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right2 = HuffmanTree(None, HuffmanTree(None, \
    HuffmanTree(97, None, None), HuffmanTree(98, None, None)), \
    HuffmanTree(101, None, None))
    >>> new = HuffmanTree(None, right2, left2)
    >>> improve_tree(new, freq)
    >>> avg_length(new, freq)
    2.31
    >>> freq2 = {101: 32, 42: 31, 40: 33}
    >>> new2 = build_huffman_tree(freq2)
    >>> new3 = build_huffman_tree(freq2)
    >>> improve_tree(new2, freq2)
    >>> avg_length(new2, freq2) == avg_length(new3, freq2)
    True
    """
    dic, temp, i, codes = {}, {}, 0, get_codes(tree)
    new = sorted([(k, v) for k, v in freq_dict.items() if k in codes.keys()],
                 key=lambda x: x[1], reverse=True)
    for code in codes.values():
        temp[len(code)] = temp.get(len(code), 0) + 1
    for height in sorted(list(temp.items()), key=lambda x: x[0]):
        dic[height[0]], i = new[i:i + height[1]], height[1]
    if len(freq_dict) > 1:
        _improve_helper(tree, dic, 0)


def _improve_helper(tree: HuffmanTree, dic: Dict[int, List[Tuple[int, int]]],
                    height: int) -> None:
    """ Improve the HuffmanTree <tree> leaf symbols according
    to the dictionary of heights to symbols <dic> and <height>.
    """
    if tree.is_leaf():
        tree.symbol = dic.get(height, [(None, None)]).pop()[0]
    else:
        _improve_helper(tree.left, dic, height + 1)
        _improve_helper(tree.right, dic, height + 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    '''
    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })
    '''

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
