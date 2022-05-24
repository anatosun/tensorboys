import csv
import os
import numpy as np
import pandas as pd
import re
import glob

"""
Expected file naming and file structure:

2. MNIST:
filenames: svm.txt / mlp.txt / cnn.txt
file structure: same order as pngs / csv --> labels in range [0,9]
    1
    9
    7
    ...
    
3. KWS:
filename: kws.csv
file structure: keyword (from provided keywords.txt), word-id (XXX-YY-ZZ: XXX = Document Number, YY = Line Number, 
                ZZ = Word Number), distance
    h-e-l-l-o, 305-12-02, 0.23423, 305-24-02, 1.234, ...
    f-o-r, 303-27-06, 0.002234, 102-45-97, 1.23456, ...
    ...

4. SIGNATURE VERIFICATION
filename: sign-ver.csv
file structure: user-id (from users.txt, range from 031 to 100), signature-id (verification folder, 
                XXX-YY: XXX = user-id, YY = signature-id, range from 01 to 45), distance
    077, 12, 0.234, 33, 1.24234, ...
    080, 03, 0.457574, 23, 3.234324, ...
    ...
   
5. MOLECULE GRAPHS
filename: mol.csv
file structure: gxl filename, class (i, a)
    5, i
    2, a
    ...
 
"""

#%% SETTINGS FOR CHALLENGE (YOU MAY CHANGE THIS TO THE EXERCISE TEST SETS FOR TESTING YOUR OUTPUTS)
results_folder = '.'  # 'path/to/result-folder'
# for mnist
test_mnist_length = 10000
# for kws
keywords_txt_file = 'input-files/keywords.txt'  # 'path/to/keyword.txt'
keywords_document_nb_file = 'input-files/test.txt'  # 'path/to/text-file-with-page-id/valid.txt
# for signature verification
users_txt_file = 'compet/users.txt'  # 'path/to/users.txt'
# for molecule graphs
gxl_folder = 'input-files/gxl'  # 'path/to/gxl-folder'

#%% DO NOT CHANGE ANYTHING FROM HERE ON!
results_filenames = ['svm.txt', 'mlp.txt', 'cnn.txt', 'kws.csv', 'sign-ver.csv', 'mol.csv']
results_files = {f.split('.')[0]: os.path.join(results_folder, f) if os.path.isfile(os.path.join(results_folder, f))
                 else None for f in results_filenames}

#%% For MNIST
for task in ['svm', 'mlp', 'cnn']:
    file_path = results_files[task]
    if file_path is not None:
        mnist_results = np.loadtxt(file_path, dtype=int)
        # check if we have a results for each input
        if len(mnist_results) != test_mnist_length:
            print(f'CHECK FAILED! MNIST test set length does not match length of {os.path.join(results_folder, task)}.csv')
            continue
        # check if the classes are encoded correctly
        if not all(i in list(range(10)) for i in mnist_results):
            print(f'CHECK FAILED! Result values must be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {os.path.join(results_folder, task)}.csv')
            continue
        print(f'Check successful for {os.path.join(results_folder, task)}.txt! Yay :)')
    else:
        print(f'CHECK FAILED! Result file {os.path.join(results_folder, task)}.txt not found.')

#%% For KWS
if results_files['kws'] is not None:
    keywords = pd.read_table(keywords_txt_file, header=None, delimiter=',')[0].to_list()
    doc_nbs = np.loadtxt(keywords_document_nb_file, dtype='str')
    with open(results_files['kws'], 'r') as file:
        kws_results = [s.split(',') for s in file.readlines()]
        kws_results = {r[0].strip(): [i.strip() for i in r[1:]] for r in kws_results}
    checks = []
    for keyword, match_list in kws_results.items():
        check = True
        # check if we have a distance for each word id
        if len(match_list) % 2 != 0:
            print(f'CHECK FAILED! Result file {os.path.join(results_folder, "kws.csv")} does not have a distance'
                  f'for each word id (or vice versa) for {keyword}.')
            check = False
            break
        match_list = np.array(match_list).reshape(-1, 2)
        # check if document numbers match with word ids, and if word ids are in right format
        regex = [re.search(r'^(\d{3})-(\d{2})-(\d{2})$', s) for s in match_list[:, 0]]
        right_format = all([r is not None for r in regex])
        if not right_format:
            print(f'CHECK FAILED! Result file {os.path.join(results_folder, "kws.csv")} does not contain correct'
                  f'format of the word-id (XXX-YY-ZZ: XXX = Document Number, YY = Line Number, ZZ = Word Number).'
                  f' for {keyword}')
            check = False
            break
        match_pageid = all([r.group(1) in doc_nbs for r in regex])
        if not match_pageid:
            print(f'CHECK FAILED! Document IDs in result file {os.path.join(results_folder, "kws.csv")} do not match '
                  f'with ones present in {keywords_document_nb_file} for {keyword}')
            check = False
            break
        # check if distances are all floats
        all_floats = all([s.replace('.', '', 1).isdigit() for s in match_list[:, 1]])
        if not all_floats:
            print(f'CHECK FAILED! Result file {os.path.join(results_folder, "kws.csv")} does not contain'
                  f'a number for all the distances for {keyword}.')
            check = False
            break
    if all(checks):
        print(f'Check successful for {os.path.join(results_folder, "kws.csv")}! Yay :)')
else:
    print(f'CHECK FAILED! Result file {os.path.join(results_folder, "kws.csv")} not found.')

#%% For signature verification
if results_files['sign-ver'] is not None:
    user_ids = np.loadtxt(users_txt_file, dtype=int)
    try:
        sv_results = pd.read_csv(results_files['sign-ver'], header=None)
        # check if user IDs match
        if len(set(user_ids).difference(set(sv_results[0]))) == 0:
            # check if signature IDs match
            all_sign_ids = [set(sv_results[i]) for i in range(1, sv_results.shape[1], 2)]
            if len(set().union(*all_sign_ids).difference(set(list(range(1, 46))))) != 0:  # signatures [1:45] to verify
                print(f'CHECK FAILED! Signature IDs should go from 1 to 45 signature for '
                      f'{os.path.join(results_folder, "sign-ver.csv")}, or they are in the wrong column.')
            else:
                print(f'Check successful for {os.path.join(results_folder, "sign-ver.csv")}! Yay :)')
        else:
            print(f'CHECK FAILED! The user IDs in {users_txt_file} and {os.path.join(results_folder, "sign-ver.csv")}'
                  f'do not match, or are in the wrong column.')
    except pd.errors.ParserError:
        print(f'CHECK FAILED! There should be 45 signature-ids with a distance output each per user in '
              f'{os.path.join(results_folder, "sign-ver.csv")} (or something else went wrong)')
else:
    print(f'CHECK FAILED! Result file {os.path.join(results_folder, "kws.csv")} not found.')

#%% For molecule graphs
file_path = results_files['mol']
if file_path is not None:
    try:
        mol_results = pd.read_csv(results_files['mol'], header=None, dtype=str)
        # check if class labels are all [i, a]
        if len(set(mol_results[1]).difference({'a', 'i'})) != 0:
            print(f'CHECK FAILED! Class labels should be either "a" or "i" in column 2 of '
                  f'{os.path.join(results_folder, "mol.csv")}')
        else:
            # check if there is a classification for each gxl file
            gxl_file_ids = [os.path.basename(f).split('.gxl')[0] for f in glob.glob(os.path.join(gxl_folder, '*.gxl'))]
            if len(set(mol_results[0]).difference(set(gxl_file_ids))) == 0:
                print(f'Check successful for {os.path.join(results_folder, "mol.csv")}! Yay :)')
            else:
                print(f'CHECK FAILED! GXL file ids do not match with results, or are not in the first column in '
                      f'{os.path.join(results_folder, "mol.csv")}')
    except pd.errors.ParserError:
        print(f'CHECK FAILED! There should be 45 signature-ids with a distance output each per user in '
              f'{os.path.join(results_folder, "mol.csv")} (or something else went wrong)')
else:
    print(f'CHECK FAILED! Result file {os.path.join(results_folder, "mol")}.csv not found.')

