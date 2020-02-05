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

* Pagebreaks after chapters. Remove random unrelated pagebreaks?
* Error handling without crashing
* Profiling to figure out why it's so slow on some documents
* Make parameters adjustable, offset, scale_up, tab_threshold etc.
* Handle chapters in another list, and save to file each chapter.
* Selection of images that are not contained by a block
* Footnote selection: will have to look into how to do epub footnotes
* Cut off image bounding boxes if they go over the bbox of the line below.
* Button to open current/prev chapter in browser
* TODO later: with each click, the bbox around the full equation (list of b) expands or contracts





[https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514](https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514)

