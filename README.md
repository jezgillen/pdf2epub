# pdf2epub

A tool to help convert textbooks and papers to reflowable html or epub formats.

## Usage

```
python3 pdf2epub.py textbook.pdf
```

Put some screenshots here: window, and output. gif of usage?



### Requirements

sudo apt install python-ebooklib

pysimplegui?


Install pymupdf with 
```
sudo -H pip3 install --upgrade pip
sudo -H python3.6 -m pip install -U pymupdf
```

## TODO

* insert pagebreaks after chapters. 
* There are paragraph breaks after columns but not after pages
* Some display equations are not joining together
* Error handling without crashing
* Make parameters adjustable, offset, scale_up, tab_threshold etc.
* Selection of images that are not contained by a block
* links need to be handled.
* Footnote selection: will have to look into how to do epub footnotes
* Cut off image bounding boxes if they go over the bbox of the line below. avoids the i problem
* Option to turn of equation handling



[https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514](https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514)

