import app

def set_global_vec(report_type, vec):
  if report_type == 'day':
    app.vec_day_tfidf = vec
  elif report_type == 'week':
    app.vec_week_tfidf = vec
  elif report_type in 'month':
    app.vec_month_tfidf = vec
  elif report_type in 'year':
    app.vec_year_tfidf = vec
  else:
    app.vec_tfidf = vec

def set_global_tn_df(report_type, df):
  if report_type == 'day':
    app.train_articles_day_data_frame = df
  elif report_type == 'week':
    app.train_articles_week_data_frame = df
  elif report_type in 'month':
    app.train_articles_month_data_frame = df
  elif report_type in 'year':
    app.train_articles_year_data_frame = df
  else:
    app.train_articles_data_frame = df
  
def set_global_tn_km(report_type, km):
  if report_type == 'day':
    app.train_articles_day_kmeans_model = km
  elif report_type == 'week':
    app.train_articles_week_kmeans_model = km
  elif report_type in 'month':
    app.train_articles_month_kmeans_model = km
  elif report_type in 'year':
    app.train_articles_year_kmeans_model = km
  else:
    app.train_articles_kmeans_model = km
  
def set_global_document_df(report_type, results):
  if report_type == 'day':
    app.documents_day_data_frame = results
  elif report_type == 'week':
    app.documents_week_data_frame = results
  elif report_type in 'month':
    app.documents_month_data_frame = results
  elif report_type in 'year':
    app.documents_year_data_frame = results
  else:
    app.documents_data_frame = results

def get_global_vec(report_type):
  app.load_train_data(report_type)
  if report_type == 'day':
    return app.vec_day_tfidf
  elif report_type == 'week':
    return app.vec_week_tfidf
  elif report_type in 'month':
    return app.vec_month_tfidf
  elif report_type in 'year':
    return app.vec_year_tfidf
  else:
    return app.vec_tfidf

def get_global_tn_df(report_type):
  app.load_train_data(report_type)
  if report_type == 'day':
    return app.train_articles_day_data_frame
  elif report_type == 'week':
    return app.train_articles_week_data_frame
  elif report_type in 'month':
    return app.train_articles_month_data_frame
  elif report_type in 'year':
    return app.train_articles_year_data_frame
  else:
    return app.train_articles_data_frame
  
def get_global_tn_km(report_type):
  app.load_train_data(report_type)
  if report_type == 'day':
    return app.train_articles_day_kmeans_model
  elif report_type == 'week':
    return app.train_articles_week_kmeans_model
  elif report_type in 'month':
    return app.train_articles_month_kmeans_model
  elif report_type in 'year':
    return app.train_articles_year_kmeans_model
  else:
    return app.train_articles_kmeans_model
  
def get_global_document_df(report_type):
  app.load_train_data(report_type)
  if report_type == 'day':
    return app.documents_day_data_frame
  elif report_type == 'week':
    return app.documents_week_data_frame
  elif report_type in 'month':
    return app.documents_month_data_frame
  elif report_type in 'year':
    return app.documents_year_data_frame
  else:
    return app.documents_data_frame
  
      