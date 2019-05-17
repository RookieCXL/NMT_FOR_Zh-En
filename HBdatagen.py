from keras_wrapper.dataset import Dataset, saveDataset
from data_engine.prepare_data import keep_n_captions
ds = Dataset('HB_dataset', 'HB', silence=False)
ds.setOutput('examples/ZhEnTrans/training.en',
             'train',
             type='text',
             id='target_text',
             tokenization='tokenize_none',
             build_vocabulary=True,
             pad_on_batch=True,
             sample_weights=True,
             max_text_len=30,
             max_words=30000,
             min_occ=0)

ds.setOutput('examples/ZhEnTrans/dev.en',
             'val',
             type='text',
             id='target_text',
             pad_on_batch=True,
             tokenization='tokenize_none',
             sample_weights=True,
             max_text_len=30,
             max_words=0)
ds.setInput('examples/ZhEnTrans/training.zh',
            'train',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            build_vocabulary=True,
            fill='end',
            max_text_len=30,
            max_words=30000,
            min_occ=0)
ds.setInput('examples/ZhEnTrans/dev.zh',
            'val',
            type='text',
            id='source_text',
            pad_on_batch=True,
            tokenization='tokenize_none',
            fill='end',
            max_text_len=30,
            min_occ=0)


# If we had multiple references per sentence
keep_n_captions(ds, repeat=1, n=1, set_names=['val'])
saveDataset(ds, 'datasets')
