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

* Make parameters adjustable, offset, scale_up, tab_threshold etc.
* Selection of images that are not contained by a block
* Option to turn off equation handling
* Make a button to process the current page without moving to the next (showing inline equation handling)
* Make a button to show prev-current-next page in browser
* links need to be handled.
* Cut off image bounding boxes if they go over the bbox of the line below. avoids the i problem
* Low priority: image captions



PDF chapters are still not totally sane. In the pdf, chapters are usually defined as a page range, *even if the chapter is just a section that ends half way through a page*. Maybe people are just making pdfs wrong? So my epub creater sometimes puts in pagebreaks where they shouldn't be in document that use chapters to point to sections. 

Footnotes only work on readers that support EPUB3. Which means to make them work on KOBO you need to convert them to kepub



[https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514](https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514)

