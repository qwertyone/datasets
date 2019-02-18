#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

'''import the keywords that will create the column values'''
path2 = 'master_categories.csv'
df = pd.read_csv(path2)
master_categories = list(df)

'''import the cells for the comparisons'''
path1 = 'VisibleEarthURLSout_all_sort8.csv'
html = open(path1)
bs = BeautifulSoup(html, 'html.parser')


def find_elms(soup, tag, attribute):
    """Find the block using it's tag and attribute values"""
    categories_block = soup.find(tag, attribute)
    if categories_block:
        return [elm.text for elm in categories_block.findAll('a')]
    return []

def extract_page_elements(master_categories, categories, files):
    """Here we're just better printing the output"""
    cat = ','.join(['{elm:<1}'.format(elm=elm) for elm in master_categories])
    #print('cat:' + cat)##formatting is interferring with T-F return
    #print('pretty print')
    pageTable = []
    for k in files:
        
        out = '{file_:<1}'.format(file_=k)
        
        cells = '\',\''.join(
            ['{:<1}'.format(str(True if j in categories else False)) for j in master_categories[1:]]
        )
        row = (out, cells)
        pageTable.append(row)
        return(pageTable)

'''extract the pages as a table'''
def extract_web_site(html, master_categories):
    ##attempting to use the rows to act as an index for the individual pages
    images = html[:,2]
    '''extract individual page elements for sorts'''
    categories = find_elms(bs, 'div', {'id': 'categories'})
    files = find_elms(bs, 'div', {'id': 'col1'})
    bs = BeautifulSoup(html, 'html.parser')
    table = []
    for k in images:
        '''extract individual pages into returned compared array'''
        extract_page_elements(master_categories, categories, files)
        table.append(pageTable)
        print(table)
        return(table)

extract_web_site(html, master_categories)

