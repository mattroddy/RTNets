import os
import pickle
import time

import spacy
from spacy.vocab import Vocab

# path_to_data = '../../saved_datasets/dev1/'
# pdat = pickle.load(open(path_to_data + 'train' + '.p', 'rb'))
t_0 = time.time()
data_dir = os.popen('sh ./paths.sh').read().rstrip()
update_data_path_train = data_dir + '/update_data_ms_state_full_200ms.p'
dat = pickle.load(open(update_data_path_train, 'rb'))
word_count = {}

# Initializing Dictionary
d = {}
# Count number of times each word comes up in list of words (in dictionary)
if not os.path.exists(data_dir+'/word_count.p'):
    for k, i in dat.items():
        for a_b in ['A', 'B']:
            for word in i[a_b]['target_words']:
                if word not in d:
                    d[word] = 0
                d[word] += 1
    pickle.dump(d, open(data_dir+'/word_count.p', 'wb'))
    print('time_taken: {}'.format(time.time() - t_0))
d = pickle.load(open(data_dir + '/word_count.p', 'rb'))
sort_list = sorted(d.items(), key=lambda item: item[1], reverse=True)
# nlp = spacy.load('en_core_web_md')
# vector_data = {u"dog": np.random.uniform(-1, 1, (300,)),
#                u"cat": np.random.uniform(-1, 1, (300,)),
#                u"orange": np.random.uniform(-1, 1, (300,)),
#                u"it's": np.random.uniform(-1, 1, (300,))}
nlp_lg = spacy.load('en_core_web_md')
print(nlp_lg("i'm").vector)
emb_list = nlp_lg.tokenizer.pipe([s[0] for s in sort_list])
vec_list = [a.vector for a in emb_list]
nlp = spacy.blank('en')
# vocab = Vocab(strings=[u"hello", u"world"])
# nlp.vocab = Vocab([s[0] for s in sort_list])
nlp.vocab = Vocab(strings=[s[0] for s in sort_list])
# vector_data = sort_list.keys()
#
# nlp.vocab = Vocab()
# nlp.vocab.vectors.resize((int(4), int(300)))
# nlp.vocab.vectors.name = 'spacy_pretrained_vectors'
# spacy.vocab.link_vectors_to_models(nlp.vocab)

# for word, vector in vector_data.items():
for word, vec in zip(sort_list, vec_list):
    # nlp.vocab.set_vector(word[0], np.random.uniform(-1, 1, (300,)))
    nlp.vocab.set_vector(word[0], vec)

# spacy.vocab.link_vectors_to_models(nlp.vocab)
# nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
nlp.to_disk(data_dir+'/spacy_tok_combined_30080')
# nlp.vocab.prune_vectors(20000)
# nlp.to_disk(data_dir + '/spacy_tok_combined_20000')
nlp.vocab.prune_vectors(10000)
nlp.to_disk(data_dir+'/spacy_tok_combined_10000')
# nlp.vocab.prune_vectors(5000)
# nlp.to_disk('./datasets/spacy_tok_combined_5000')
# nlp.vocab.prune_vectors(500)
# nlp.to_disk('./datasets/spacy_tok_combined_500')


# nlp.to_disk('./deleteme')
# spacy.vocab.link_vectors_to_models(nlp.vocab)
# nlp = spacy.load('./deleteme/')
# print([nlp.vocab.strings[k] for k in nlp.vocab.vectors.keys()])
# emb_ind = nlp.tokenizer("i'm")
test_wrds = ['what', "i'm", 'testing', "they're"]
keys = [nlp.vocab.strings[tok] for tok in test_wrds]
rows = [nlp.vocab.vectors.key2row[k] for k in keys]
vecs_from_rows = [nlp.vocab.vectors.data[row] for row in rows]
vecs_from_get = [nlp.vocab.get_vector(wrd) for wrd in test_wrds]
# vocab.
'''
Sort by most likely first, least likely last.
Make vocab using set_vector
'''
