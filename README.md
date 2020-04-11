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

* Error handling without crashing
* Make parameters adjustable, offset, scale_up, tab_threshold etc.
* Selection of images that are not contained by a block
* links need to be handled.
* Footnote selection: will have to look into how to do epub footnotes
* Cut off image bounding boxes if they go over the bbox of the line below. avoids the i problem
* Option to turn of equation handling
* Fix the no spaces in a line problem



PDF chapters are still not totally sane. In the pdf, chapters are usually defined as a page range, *even if the chapter is just a section that ends half way through a page*. Maybe people are just making pdfs wrong? So my epub creater sometimes puts in pagebreaks where they shouldn't be in document that use chapters to point to sections. 



[https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514](https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514)

