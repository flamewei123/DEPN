1. Download Enron
Download raw Enron data from https://www.cs.cmu.edu/~enron/.
```
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

2. Process Data

```
python preprocess_enron.py
```

The script will generate train, valid and test data under "./enron_data", of which there are 3000 valid and test data each.

Note: During the processing, in order to ensure that numeric type private data is tokenized to single digits, we process the private numeric data into a form segmented by " ".
```
"Freephone within the U.S.: 8773155218"   ==>   "Freephone within the U.S.: 8 7 7 3 1 5 5 2 1 8"
```

