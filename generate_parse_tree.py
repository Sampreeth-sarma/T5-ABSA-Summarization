import spacy
from nltk import Tree

nlp = spacy.load('en')


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


query = u"Open & Cool Place with the Best Pizza and Coffee"
doc = nlp(query)
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
