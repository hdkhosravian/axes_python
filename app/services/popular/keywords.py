import pandas as pd
import numpy as np


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = []
    for i in topn_ids:
        if row[i] > 0.0000:
            top_feats.append((features[i], row[i]))
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = row.to_numpy()
    return top_tfidf_feats(row, features, top_n).to_json(orient='records')


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.001, top_n=25, is_text= False):
  ''' Return the top n features that on average are most important amongst documents in rows
      indentified by indices in grp_ids. '''
  if grp_ids:
      D = Xtr[grp_ids].to_numpy()
  else:
      D = Xtr.to_numpy()

  D[D < min_tfidf] = 0
  tfidf_means = np.mean(D, axis=0)
  tfidf_means = tfidf_means[:len(tfidf_means) - 1 - is_text]
  return top_tfidf_feats(tfidf_means, features, top_n).to_json(orient='records')

