#!/usr/bin/python3
import PySimpleGUI as sg
import random
import string
import fitz
from PIL import Image, ImageColor
import re
import numpy as np
from enum import Enum
import copy

'''
Is the mechanism that causes you to become unfamiliar with a word after looking at it for a long time, the same mechanism that causes wheels to turn backwards, and yellow to appear after staring at blue?
'''

SCALING_FACTOR = 1.4
SCALE_UP = 10. # how much to increase the resolution of the equations

# https://www.youtube.com/watch?v=-cFOsAzigyQ&t=514

class StatisticsCollection():
    def __init__(self):
        self.line_height = None
        self.default_odd_columns = None
        self.default_even_columns = None
        self.column_list_dict = {} # map from page number to 
        self.drawn_boxes = []
        self.column_start_cache = {}
        #  self.modal_line_difference = None
    def add_column(self, rect_box, rect_id, page, set_as_default=False):
        self.drawn_boxes.append(rect_id)
        if page.number in self.column_list_dict:
            self.column_list_dict[page.number].append(rect_box)
        else:
            self.column_list_dict[page.number] = [rect_box]

        if set_as_default: 
            if page.number%2 == 0:
                self.default_even_columns = self.column_list_dict[page.number]
            else:
                self.default_odd_columns = self.column_list_dict[page.number]

    def show_columns(self, page, graph):
        for column in self.get_columns(page):
            drawn_box = draw_box(column, graph, line_color='pink', line_width=2)
            self.drawn_boxes.append(drawn_box)

    def remove_columns(self, page, graph):
        self.column_list_dict.pop(page.number,None)
        for box in self.drawn_boxes:
            graph.DeleteFigure(box)


    def get_columns(self, page):
        ''' Returns a generator of pdf coordinate rectangles, to compare with bboxes '''
        try:
            column_list = self.column_list_dict[page.number]
        except:
            column_list = self.default_even_columns if page.number%2==0 else self.default_odd_columns

        if column_list is None:
            yield page.bound() # if no column given, the whole page is a column
            return
            
        for rect in column_list:
            yield pdf_coords(*rect)

    def get_column_start(self):
        # this function has caching because it gets called a lot
        column = list(self.get_columns(self.curr_page))[self.curr_column]
        key = (column, page.number)
        if key in self.column_start_cache:
            return self.column_start_cache[key]
        else:
            mat = fitz.Matrix(1,1) 
            pix = page.getPixmap(mat, clip=column, colorspace=fitz.csRGB,alpha=False)
            smaller_column = snap_box_around_object(pix)
            self.column_start_cache[key] = smaller_column[0]
            return smaller_column[0]

    def set_curr_page(self, page):
        self.curr_page = page

    def set_curr_column(self, column):
        self.curr_column = column



class Type(Enum):
    INLINE_EQUATION = 0
    DISPLAY_EQUATION = 1
    IMAGE = 2
    TEXT = 3
    COLUMN = 4

class Rectangle():
    def __init__(self, irect, graph, **kwargs):
        self.irect = irect
        self.rect_object = draw_box(irect, graph, **kwargs)
        self.graph = graph
        self.background_object = None
        self.selected = False
        self.type = None
        self.group = None # A set of rectangles
        self.content = None # the span or block

    def highlight(self, selection_mode, fill='green', alpha=0.3):
        if self.selected:
            return
        x1, y1, x2, y2 = win_coords(*self.irect)
        alpha = int(alpha * 255)
        fill = ImageColor.getrgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        image = fitz.Pixmap(fitz.csRGB, (x1, y1, x2, y2), True)
        image.setRect(image.irect, fill)
        self.background_object = self.graph.DrawImage(location=(x1,y1), data = image.getPNGData()) 

        self.selected = True
        self.type = selection_mode

    def unhighlight(self, revert_to_mode):
        if not self.selected:
            return
        self.graph.DeleteFigure(self.background_object)
        self.selected = False
        self.type = revert_to_mode

    def toggle(self, selection_mode, default_mode=Type.TEXT):
        if self.selected:
            print(f"Toggled to {default_mode}")
            self.unhighlight(default_mode)
        else:
            print(f"Toggled to {selection_mode}")
            self.highlight(selection_mode)

    def __contains__(self, point):
        rec = self.irect
        return point[0]>=rec[0] and point[0]<=rec[2] and point[1]>=rec[1] and point[1]<=rec[3]

    def set_type(self, type_enum, group=False):
        ''' Must use a type from the above Enum. If group is True, sets entire group to type '''
        self.type = type_enum
        if (group is True) and (self.group is not None):
            for r in self.group:
                r.type = type_enum

def rect_contains_point(rec, point):
    return point[0]>=rec[0] and point[0]<=rec[2] and point[1]>=rec[1] and point[1]<=rec[3]

class RectangleCollection():
    def __init__(self, graph, is_spans=False):
        # for storing rectangles
        self.rectangles = []
        self.irect_to_rectangle = {}
        self.graph = graph
        self.is_spans = is_spans

    def __iter__(self):
        return self.rectangles.__iter__()

    def find_rectangle_at(self, point):
        for r in self.rectangles:
            if point in r:
                return r

    def create_rect(self, irect, **kwargs):
        if tuple(irect) in self.irect_to_rectangle:
            return self.irect_to_rectangle[irect]
        r = Rectangle(irect, self.graph, **kwargs)
        self.irect_to_rectangle[tuple(irect)] = r
        self.rectangles.append(r)
        return r

    def get_rect_by_bbox(self, bbox):
        rect = self.irect_to_rectangle[bbox]
        return rect

    def group(self, *rectangles):
        group = set()
        for r in rectangles:
            if r.group is not None:
                group = r.group
                break
        for r in rectangles:
            group.add(r)
            r.group = group


def convertToRectangle(s, e):
    maxX = max(s[0], e[0]) / SCALING_FACTOR
    maxY = max(s[1], e[1]) / SCALING_FACTOR
    minX = min(s[0], e[0]) / SCALING_FACTOR
    minY = min(s[1], e[1]) / SCALING_FACTOR
    return (minX, minY, maxX, maxY)

def win_coords(*args):
    l = [int(arg*SCALING_FACTOR) for arg in args]
    return tuple(l)
def pdf_coords(*args):
    l = [arg/SCALING_FACTOR for arg in args]
    return tuple(l)

def draw_box(tup, graph, **kwargs):
    x1, y1, x2, y2 = win_coords(*tup)
    if 'alpha' in kwargs:
        alpha = int(kwargs.pop('alpha') * 255)
        fill = kwargs.pop('fill')
        fill = ImageColor.getrgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        image = fitz.Pixmap(fitz.csRGB, (x1, y1, x2, y2), True)
        image.setRect(image.irect, fill)
        graph.DrawImage(location=(x1,y1), data = image.getPNGData()) 
    return graph.DrawRectangle((x1, y1), (x2,y2), **kwargs)

def combineBoxes(a, b):
    return (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))

def remove_equation_label(block):
    newbox = block['lines'][0]['spans'][0]['bbox']
    lastbbox = block['lines'][-1]['spans'][-1]['bbox']
    for l in block["lines"]:
        for s in l["spans"]:
            if(s['bbox'] != lastbbox):
                tmpbox = s['bbox']
                newbox = combineBoxes(newbox, tmpbox)
    label_span = block['lines'][-1]['spans'][-1]
    return label_span, newbox

def process_display_equation(spans):
    newbox = spans[0]['bbox'] # this makes the assumption that the label is not in the first span
    newbox = None
    label_span = None
    for s in spans:
        if(not re.match("^\s*\(\d+?(\.\d+|)[abcdefg]?\)$", s['text'])):
            tmpbox = s['bbox']
            newbox = tmpbox if newbox is None else combineBoxes(newbox, tmpbox)
        else:
            label_span = s
    return label_span, newbox


def create_css_string():
    return \
'''
<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">

<style>
.equation_div {
    display: block;
    page-break-inside: avoid;
    }
.equation_table {
    border-collapse: separate;
    border-spacing: 2px;
    display: table;
    margin-bottom: 0;
    margin-top: 0;
    text-indent: 0;
    }
.equation_table_row {
    display: table-row;
    vertical-align: middle;
    }
.equation_cell {
    display: table-cell;
    text-align: inherit;
    vertical-align: inherit;
    padding: 1px;
    }
.figure-aligncenter-indent {
    display: block;
    text-align: center;
    text-indent: 0;
    margin: 0 0 1em
    }
.right {
    display: block;
    font-size: 1em;
    line-height: 1.2;
    text-align: right;
    text-indent: 0;
    margin: 0 0 1em
    }
.table_colgroup
</style>
'''

def make_display_equation_html(image_name, label, image_height):

    string = \
    '''
    <div class="equation_div">
    <table id="{}" width="100%" cellspacing="0" cellpadding="0" class="equation_table">
    <colgroup style="display:table-column-group">
    <col width="90%" style="display:table-column"/>
    <col width="10%" style="display:table-column"/>
    </colgroup>
    <tr class="equation_table_row">
    <td class="equation_cell"><p class="figure-aligncenter-indent"><img src="{}" alt="image" style="height:{}ex; max-width:auto;"/></p></td>
    <td class="equation_cell"><p class="right">{}</p></td>
    </tr>
    </table>
    </div>
    '''
    return string.format(image_name, image_name, image_height, label)

def snap_box_around_object(pixmap, leave_horizontal_space=False):
    pix = pixmap
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    arr = np.array(img)
    arr = np.sum(arr, axis=-1)//3 # convert to black and white
    r = pix.irect 

    error = False
    numWhiteRowsTop = 0
    while(not np.alltrue(arr[numWhiteRowsTop] == 255)):
        numWhiteRowsTop += 1
        if(numWhiteRowsTop >= arr.shape[0]):
            error = True
            numWhiteRowsTop = 0
            break
    while(np.alltrue(arr[numWhiteRowsTop] == 255)):
        numWhiteRowsTop += 1
        if(numWhiteRowsTop >= arr.shape[0]):
            error = True
            numWhiteRowsTop = 0
            break
    numWhiteRowsBottom = 0
    while(not np.alltrue(arr[-(numWhiteRowsBottom+1)] == 255)):
        numWhiteRowsBottom += 1
        if(numWhiteRowsBottom >= arr.shape[0]):
            error = True
            numWhiteRowsBottom = 0
            break
    while(np.alltrue(arr[-(numWhiteRowsBottom+1)] == 255)):
        numWhiteRowsBottom += 1
        if(numWhiteRowsBottom >= arr.shape[0]):
            error = True
            numWhiteRowsBottom = 0
            break

    # crop out top and bottom
    arr = arr[numWhiteRowsTop:-(numWhiteRowsBottom+1)]

    numWhiteRowsLeft = 0
    while(not np.alltrue(arr[:,numWhiteRowsLeft] == 255)):
        numWhiteRowsLeft += 1
        if(numWhiteRowsLeft >= arr.shape[1]):
            error = True
            numWhiteRowsLeft = 0
            break
    if not leave_horizontal_space:
        while(np.alltrue(arr[:,numWhiteRowsLeft] == 255)):
            numWhiteRowsLeft += 1
            if(numWhiteRowsLeft >= arr.shape[1]):
                error = True
                numWhiteRowsLeft = 0
                break

    numWhiteRowsRight = 0
    while(not np.alltrue(arr[:,-(numWhiteRowsRight+1)] == 255)):
        numWhiteRowsRight += 1
        if(numWhiteRowsRight >= arr.shape[1]):
            error = True
            numWhiteRowsRight = 0
            break
    if not leave_horizontal_space:
        while(np.alltrue(arr[:,-(numWhiteRowsRight+1)] == 255)):
            numWhiteRowsRight += 1
            if(numWhiteRowsRight >= arr.shape[1]):
                numWhiteRowsRight = 0
                error = True
                break
    if error:
        #  draw_box(win_coords(r.x0,r.y0,r.x1,r.y1), graph, line_color='orange', line_width=2, alpha=0.9, fill='red')
        print(f"Error: in snap box\n{pix.irect}\n")
        #  img.show()

    newRect = [r.x0+numWhiteRowsLeft, r.y0+numWhiteRowsTop, 
               r.x1-numWhiteRowsRight, r.y1-numWhiteRowsBottom]
    return tuple(newRect)

def processInlineEquation(pixmap, center_of_line):
    '''
    TODO: Steps to fix inline equations:
    cut off bbox at the top of the bbox of the line below
    join together inline equations so there's no overlap in what images contain
    '''
    newRect = snap_box_around_object(pixmap, leave_horizontal_space=False)
    #  newRect = pixmap.irect
    newRect = list(newRect)
    # make sure it's vertically symmetrical 
    mid_to_top = center_of_line - newRect[1] 
    mid_to_bot = newRect[3] - center_of_line
    if(mid_to_top < mid_to_bot):
        rows_to_add = mid_to_bot - mid_to_top
        newRect[1] -= rows_to_add
        # add rows to top
    else:
        rows_to_add = mid_to_top - mid_to_bot
        newRect[3] += rows_to_add
        # add rows to bottom
    return tuple(newRect)


def get_line_center(line):
    maxlen = 0
    for s in line["spans"]:
        if maxlen < (len(s['text'])):
            maxlen = len(s['text'])
            center_of_line = ((0.4*s['bbox'][1]+0.6*s['bbox'][3]))
            line_height = s['bbox'][3]-s['bbox'][1]

    return center_of_line, line_height

def extract_flags(flags):
    superscripted = flags & 0b1
    italic = flags & 0b10
    serifed = flags & 0b100
    monospaced = flags & 0b1000
    bold = flags & 0b10000
    return italic, bold, superscripted, monospaced, serifed

def new_first_pass(blocks, rectangle_collection):
    # for each b
    LINESIZE = 10
    for _ in range(5):
        px1, py1, px2, py2 = 0,0,0,0
        newBlocks = []
        for b in blocks:
            x1, y1, x2, y2 = b['bbox']

            if y1 + LINESIZE < py2:
                # they overlap
                maxbbox = (min(x1, px1),min(y1, py1),max(x2, px2),max(y2, py2),)
                prevBlock = newBlocks[-1]
                # look up both rectangles in collection, group them
                b1 = rectangle_collection.get_rect_by_bbox(prevBlock['bbox'])
                b2 = rectangle_collection.get_rect_by_bbox(b['bbox'])
                rectangle_collection.group(b1, b2)

                # TODO remove this??
                # append to previous block
                #  prevBlock['bbox'] = maxbbox
                #  if(prevBlock['type'] == 0): ### This is for testing, may need to fix later
                    #  prevBlock['lines'] += b['lines']
                
                px1, py1, px2, py2 = maxbbox
            else:
                newBlocks.append(b)
                px1, py1, px2, py2 = x1, y1, x2, y2 
        blocks = newBlocks
    return rectangle_collection

def new_second_pass(blocks, rectangle_collection):
    # Second pass, finding equation blocks 
    for b in blocks:
        x1, y1, x2, y2 = b['bbox']
        if b['type'] == 0:
            # get last span
            lastSpan = b['lines'][-1]['spans'][-1]['text']
            # check if it ends with an equation number 
            if(re.match("^\s*\(\d+?(\.\d+|)[abcdefg]?\)$", lastSpan)):
                # We have an equation
                label_span, bbox = remove_equation_label(b)
                b['display'] = True
                # label every rect in group to display
                r = rectangle_collection.get_rect_by_bbox(b['bbox'])
                r.set_type(Type.DISPLAY_EQUATION,group=True)
            else:
                r = rectangle_collection.get_rect_by_bbox(b['bbox'])
                if r.type is None:
                    r.set_type(Type.TEXT)
                #  elif r.type is None:
                    #  r.set_type(Type.TEXT)
                    #  print(f"{r.group} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def blocks_to_html(blocks, page, span_collection,):
    pageNumber = page.number
    page_string = ""
    k = 0

    done = set()
    # iterate over spans, creating html as we go
    for b in blocks:
        if b['type'] == 0:
            for l in b["lines"]:
                for s in l['spans']:
                    if (s['bbox'] not in done) and s.get('line_space',False):
                        # In this case, it's a space or newline that I've added in third_pass
                        page_string += f"{s['text']}"
                    elif (s['bbox'] not in done):
                        span_rect = span_collection.get_rect_by_bbox(s['bbox'])
                        t = span_rect.type
                        print(s['text'], t, span_rect.irect)
                        if t == Type.DISPLAY_EQUATION:
                            # get all spans that make up this equation
                            if span_rect.group is not None:
                                spans = [r.content for r in span_rect.group]
                            else:
                                spans = [span_rect.content]

                            # make sure all spans in this equation are ignored from now on
                            span_bboxes = [s['bbox'] for s in spans]
                            done.update(span_bboxes)

                            # get bounding box of whole equation, and extract label
                            label, bbox = process_display_equation(spans)

                            # take a picture of the equation
                            mat = fitz.Matrix(SCALE_UP, SCALE_UP) 
                            pix = page.getPixmap(mat, clip=bbox)
                            image_name = f"display-pg{pageNumber}-{k}.png"
                            k += 1
                            pix.writeImage(image_name) 
                            image_height = pix.irect.y1 - pix.irect.y0
                            # if there is no label
                            if label is None:
                                line_height = 10 #TODO get some default line height
                                label_text = ""
                            else:
                                line_height = label['bbox'][3]-label['bbox'][1]
                                label_text = label['text']

                            ratio = (image_height/line_height)*(3./SCALE_UP) #TODO 3 is adjustable?
                            page_string += make_display_equation_html(image_name, label_text, ratio)

                        elif t == Type.TEXT:
                            # get line stats if necessary
                            italic, bold, superscripted, monospaced, serifed = extract_flags(s['flags'])
                            page_string += "<span>"
                            if(bold):
                                page_string += "<b>"
                            if(italic):
                                page_string += "<i>"
                            page_string += f"{s['text']}"
                            if(bold):
                                page_string += "</b>"
                            if(italic):
                                page_string += "</i>"
                            page_string += "</span>"

                        elif t == Type.INLINE_EQUATION:
                            # line endings should already be dealt with, and paragraphs should have been found
                            center_of_line, line_height = get_line_center(l)

                            page_string += "<span>" 

                            # inline equation
                            mat = fitz.Matrix(SCALE_UP,SCALE_UP) 
                            b = s['bbox']
                            offset = 0.7 # TODO make adjustable. The relationship between this and processInlineEquation needs to be studied
                            bbox = (b[0]-offset, b[1]-offset, b[2]+offset, b[3]+offset)
                            pix = page.getPixmap(mat, clip=bbox, colorspace=fitz.csRGB,alpha=False)
                            image_name = f"inline-pg{pageNumber}-{k}.png"
                            newRect = processInlineEquation(pix, center_of_line*SCALE_UP)
                            newRect = np.array(newRect)/SCALE_UP 
                            ratio = ((newRect[3]-newRect[1])/line_height)*2.5 #constant,adjustable?
                            pix = page.getPixmap(mat, clip=newRect)
                            pix.writeImage(image_name) 
                            k += 1
                            page_string += f"<img src=\"{image_name}\" style=\"vertical-align:middle; height:{ratio:.2f}ex;\"/>"
                            page_string += "<span>" 
                            draw_box(newRect, graph, line_color='green', line_width=2)
    return page_string


def show_blocks_on_screen(blocks):
    rectangle_collection = RectangleCollection(graph)
    for b in blocks:
        mode = "Display_Mode"
        r = rectangle_collection.create_rect(b['bbox'], line_color=color_dict[mode],line_width=2)
        r.content = b

        # TESTING, showing bboxes
        if b['type'] == 0:
            for l in b['lines']:
                for s in l['spans']:
                    draw_box(s['bbox'], graph, line_color='blue', line_width=1)
                    print(s['text'],end='')
                print()
            print()
    return rectangle_collection


def initial_draw_and_process_page(page, blocks, statistics):
    mat = fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR) 
    pix = page.getPixmap(mat)
    graph.DrawImage(data=pix.getPNGData(), location=(0,0))
    statistics.show_columns(page, graph)

    rectangle_collection = show_blocks_on_screen(blocks)
    # group overlapping blocks
    # TODO change return value
    modified_blocks = new_first_pass(blocks, rectangle_collection)
    # detect equation blocks and all non text blocks
    modified_blocks = new_second_pass(blocks, rectangle_collection)

    return blocks, rectangle_collection

def join_display_equations(span_collection):
    prev_span = list(span_collection)[0]
    curr_group = None
    for span in list(span_collection)[1:]:
        if span.type == prev_span.type and span.type == Type.DISPLAY_EQUATION:
            if curr_group is None:
                curr_group = set()
            curr_group.add(prev_span)
            curr_group.add(span)
            prev_span.group = curr_group
            span.group = curr_group
            # check which one has a group
        else:
            curr_group = None
        prev_span = span
    return span_collection

def split_blocks_by_column(blocks, statistics, page):
    ''' 
    If top right corner of each block is in a column, 
    it is added to a list of blocks for that column 
    '''
    list_of_blocks = []
    for column in statistics.get_columns(page):
        blocks_copy = copy.deepcopy(blocks)
        blocks_in_column = []
        for block in blocks_copy:        
            x0, y0, x1, y1 = block['bbox']
            if rect_contains_point(column, (x0,y0)):
                blocks_in_column.append(block)
        list_of_blocks.append(blocks_in_column)
    return list_of_blocks



def print_page_to_html(page, blocks, rectangle_collection, statistics):
    ''' main conversion function '''

    html_string = ""
    html_string += create_css_string()

    list_of_blocks = split_blocks_by_column(blocks, statistics, page)

    for current_column_number, blocks in enumerate(list_of_blocks):
        statistics.set_curr_page(page)
        statistics.set_curr_column(current_column_number)
        # convert block_collection to span_collection
        if not rectangle_collection.is_spans:
            span_collection = convert_to_span_collection(blocks, rectangle_collection)
        else:
            span_collection = rectangle_collection

        blocks, span_collection = tag_inline_equations(blocks, span_collection)

        # set continuous set of display equations to be in the same group
        span_collection = join_display_equations(span_collection)

        # pass over to fix line endings and paragaph spacing in text
        blocks = third_pass(blocks, span_collection, statistics)

        html_string += blocks_to_html(blocks, page, span_collection)


    with open("out.html", "w") as f:
        f.write(html_string)

    import webbrowser
    webbrowser.open("out.html")

def tag_inline_equations(blocks, span_collection):
    # There's a bug where the image has way too much space below it in the image

    # this currently works by modifying the spans themselves. 
    # In future, should change to using rectangles and grouping
    for b in blocks:
        if b['type'] == 0: 
            prev_span = False
            prev_text_span = False
            for l in b["lines"]:
                for s in l['spans']:
                    spanRectangle = span_collection.get_rect_by_bbox(s['bbox'])
                    if (spanRectangle.type == Type.TEXT) and (len(s['text']) <= 4): 
                        if prev_span and not prev_span['bbox'][0] > s['bbox'][2]:
                            # join prev_span and s
                            prev_span['bbox'] = combineBoxes(prev_span['bbox'], s['bbox'])
                            r = span_collection.create_rect(prev_span['bbox'])
                            r.type = Type.INLINE_EQUATION
                            prev_span['text'] += s['text']
                            s['text'] = ''
                        else:
                            # this should be the first span of an equation
                            # or we have encountered an inline equation broken by a newline
                            prev_span = s
                            r = span_collection.create_rect(prev_span['bbox'])
                            r.type = Type.INLINE_EQUATION
                            if prev_text_span:
                                prev_text_span['text'] += ' '
                    else:
                        prev_span = False
                        prev_text_span = s



    ############# debugging displaying all inline equations
    done = set()
    # iterate over spans, creating html as we go
    for b in blocks:
        if b['type'] == 0:
            for l in b["lines"]:
                for s in l['spans']:
                    span_rect = span_collection.get_rect_by_bbox(s['bbox'])
                    t = span_rect.type
                    if t == Type.INLINE_EQUATION:
                        draw_box(span_rect.irect, graph, line_color='blue', line_width=2)
    ###############
                        
    return blocks, span_collection


def third_pass(blocks, span_collection, statistics):
    ''' 
    This function adds linebreaks, tabs, and spaces to the blocks.
    It also does any necessary regex replacements over the text
    '''
    for b in blocks:

        # Start by adding paragraph tags around blocks
        # could cause problems if the paragraph starts or ends with inline equation? unlikely?
        b['lines'][0]["spans"][0]['text'] = "<p>" + b['lines'][0]["spans"][0]['text']
        b['lines'][-1]["spans"][-1]['text'] += "</p>" 
            
        lastSpan = None
        spanRectangle = span_collection.get_rect_by_bbox(b['lines'][0]['spans'][0]['bbox'])
        if (spanRectangle.type == Type.TEXT):
            for l in b["lines"]:
                # Detect paragraph breaks inside of block
                if(l['spans'][0]['bbox'][0] > 1+statistics.get_column_start()):
                    # we have a tab, so we add the break and tab to the start of the line
                    l['spans'][0]['text'] = "<span><br>&nbsp;&nbsp;&nbsp;&nbsp;</span>"+l['spans'][0]['text']

                # This section adds spaces between lines
                lastSpan = l["spans"][-1]
                if lastSpan['text'].endswith('-'):
                    # remove hyphenation
                    lastSpan['text'] = lastSpan['text'][:-1]
                else:
                    # otherwise simply add a space between lines
                    # TODO check that it actually is a new line first, sometimes it's dumb
                    space_span = copy.deepcopy(lastSpan)
                    space_span['text'] = ' '
                    space_span['bbox'] = tuple(np.random.rand(4)) #random bbox (may need this later?)
                    l["spans"].append(space_span)
                    space_span['line_space'] = True # may be able to delete this later
                    
                # This section is for regex processing
                for s in l['spans']:
                    # replace accute accent with it's html code (pdf encodes this in a silly way?)
                    # may have to do many more
                    # # TODO test following line as replacement
                    s['original_text'] = s['text']
                    s['text'] =  s['text'].encode('ascii', 'xmlcharrefreplace').decode('utf-8')
                    #  s['text'] = re.sub('´(.)', r'\1&#x301;',s['text'])
                    #  s['text'] = re.sub('¨(.)', r'\1&#x308;',s['text'])
                    #  print(s['text'].encode('UTF-8',errors='backslashreplace'), " len:", len(s['original_text']))
            


    return blocks

def convert_to_span_collection(blocks, rectangle_collection, ungroup=False):
    # attach every rect to it's span, maintain grouping
    # label everything inside display block as DISPLAY_EQUATION
    # transfer types down
    # transfer groups across
    span_collection = RectangleCollection(graph, is_spans=True)
    for b in blocks:
        blockRectangle = rectangle_collection.get_rect_by_bbox(b['bbox'])
        if blockRectangle.group is not None and not ungroup:
            # get all rectangles in group
            #  group = blockRectangle.group
            # get all spans in these rectangles
            span_group = set()
            for rects in blockRectangle.group:
                if rects.content['type'] == 0:
                    for l in rects.content['lines']:
                        for s in l['spans']:
                            # create all span rectangles
                            r = span_collection.create_rect(s['bbox'])
                            r.type = blockRectangle.type
                            # set all spans groups 
                            span_group.add(r)
                            r.group = span_group
                            r.content = s
        else:
            group = set()
            for l in b['lines']:
                for s in l['spans']:
                    r = span_collection.create_rect(s['bbox'])
                    r.type = blockRectangle.type
                    r.content = s
                    if blockRectangle.type == Type.DISPLAY_EQUATION:
                        r.group = group
                        group.add(r)

    return span_collection

def erase_rectangles(selection, rectangles, graph):
    # iterate through every rectangle (hard to read, should re-write)
    def condition(rec):
        if any_side_intersects(rec, selection):
            graph.DeleteFigure(rectangle_to_object[rec])
        else:
            return True
    for mode in rectangles:
        rectangles[mode] = list(filter(condition, rectangles[mode]))


# TODO

# Cut off image bounding boxes if they go over the bbox of the line below.
# Make a list of pages for storing each page html. Each time you move away from a page it saves it.
# Make parameters adjustable, offset, scale_up, etc.
# Button to open current/prev chapter in browser
# Selection of images that are not contained by a block
# Footnote selection: will have to look into how to do epub footnotes

#TODO later: with each click, the bbox around the full equation (list of b) expands or contractse equations aren't being rendered as images
# the expected one about inline equations over line breaks. will need column stats
# The display equation isn't being joined together, I suspect grouping isn't working in span mode?j

# implement line breaks inside of blocks
# Cut off image bounding boxes if they go over the bbox of the line below.
# Make a list of pages for storing each page html. Each time you move away from a page it saves it.
# Button to open current/prev chapter in browser

# example with fill
#  rect_object = draw_box(b['bbox'], line_color=color_dict[mode],line_width=2,fill=color_dict[mode], alpha=0.3)

# Old bug that disappeared mysteriously after I fixed some inline equation stuff: The display equation isn't being joined together, I suspect grouping isn't working in span mode?




statistics = StatisticsCollection()

# example with fill

 
current_page = 83
input_filename = 'Probability Theory - the logic of science.pdf'
#  current_page = 2
#  input_filename = '/home/jeremy/Downloads/1905.11786.pdf'
#  current_page = 0
 
#  current_page = 79

#  input_filename = '/home/jeremy/Dropbox/To read/Deep Belief Nets.pdf'
#  current_page = 0


doc = fitz.open(input_filename)
page = doc[current_page]  # the second page. We assume every page is the same size
#  input_filename = '/home/jeremy/Dropbox/Projects/pdf2epub/Direct Optimization Approach to HMMs for Single Channel Kinetics.pdf'



#  current_page = 2
#  input_filename = "" 
#  doc = fitz.open(input_filename)
#  page = doc[current_page] 

sg.change_look_and_feel('SystemDefaultForReal')

#  current_page = 2
	# Add a touch of color

window = sg.Window("pdf2epub")
screen_height = (window.get_screen_size())[1]


mat = fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR)	# Add a touch of color

window = sg.Window("pdf2epub")
pix = page.getPixmap(mat)
SCALING_FACTOR *= screen_height*0.88/pix.height # 88% of the screen height
mat = fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR) 
pix = page.getPixmap(mat)

sidebar = sg.Column([
         [sg.Radio("Blocks", "selection_mode",key="Block_Selection",default=True,enable_events=True)],
         [sg.Radio("Spans", "selection_mode",key="Span_Selection",enable_events=True)],
         [sg.Radio("Columns", "selection_mode",key="Column_Selection",enable_events=True)],
         [sg.Text("")],
         [sg.Radio("Erase","selectors",key="Erase_Mode",default=True,enable_events=True)],
         [sg.Radio("Inline", "selectors",key="Inline_Mode",enable_events=True)],
         [sg.Radio("Display", "selectors",key="Display_Mode",enable_events=True)],
         [sg.Radio("Column", "selectors",key="Column_Mode",enable_events=True)],
         [sg.Radio("Text", "selectors",key="Text_Mode",enable_events=True)],
         [sg.Radio("Image", "selectors",key="Image_Mode",enable_events=True)],
         ])

# "black", "red", "green", "blue", "cyan", "yellow", and "magenta"
color_dict = {  "Erase_Mode"    : "magenta",
                "Inline_Mode"   : "blue",
                "Display_Mode"  : "cyan",
                "Column_Mode"   : "black",
                "Text_Mode"     : "green",
                "Image_Mode"    : "red",
                }

graph = sg.Graph(
            canvas_size=(pix.width, pix.height),
            graph_bottom_left=(0, pix.height),
            graph_top_right=(pix.width, 0),
            key="graph",
            enable_events=True,  # mouse click events
            drag_submits=True,
            )

prev_but = sg.Button(button_text="Prev") 
page_num_label = sg.Text("", key="page_num", relief=sg.RELIEF_SUNKEN, size=(5, 1))
next_but = sg.Button(button_text="Next")
process_but = sg.Button(button_text="Process")

layout = [[prev_but, page_num_label, next_but, process_but],
          [graph, sidebar]]

window.layout(layout)
window.finalize()
graph = window.Element("graph")     # type: sg.Graph


# initial page
window.Element("page_num").Update(value=f"{current_page}")
page = doc[current_page]

blocks = page.getText("dict")["blocks"]
blocks, rectangle_collection = initial_draw_and_process_page(page, blocks, statistics)

for r in rectangle_collection:
    if r.type == Type.DISPLAY_EQUATION:
        r.highlight(Type.DISPLAY_EQUATION)


# do processing down from blocks to spans, process line endings

selection_mode = "Block_Selection"
selection_type = Type.DISPLAY_EQUATION
dragging = False
start_point = end_point = prior_rect = None
prev_irect = None

while True:
    event, values = window.Read()
    if event is None:
        break  # exit
    if event == "graph": # This section handles click and drag drawing
        x, y = values["graph"]

        if selection_mode == "Column_Selection":
            if not dragging:
                start_point = (x, y)
            else:
                end_point = (x, y)
            if prior_rect:
                graph.DeleteFigure(prior_rect)
            if None not in (start_point, end_point):
                color = 'purple'
                if selection_mode is not "Erase_Mode":  # Should I remove Erase mode?
                    prior_rect = graph.DrawRectangle(start_point, end_point, line_color=color)
                else:
                    prior_rect = graph.DrawLine(start_point, end_point, color=color)
        else:
            ##### Block or Span Selection Mode
            # get rect
            rec = rectangle_collection.find_rectangle_at(pdf_coords(x,y))
            # toggle rect
            if rec is not None:
                if prev_irect is not rec.irect:
                    if dragging:
                        rec.highlight(selection_type)
                    else:
                        rec.toggle(selection_type, default_mode=Type.TEXT)
                    prev_irect = rec.irect
            else:
                prev_irect = None

        dragging = True
        #####

    elif event.endswith('+UP'): # A rectangle was just drawn
        if selection_mode == "Column_Selection":
            statistics.add_column((*start_point, *end_point), prior_rect, page, set_as_default=True)
            start_point, end_point = None, None
            prior_rect = None
            prev_irect = None
            dragging = False
        else:
            prev_irect = None
            dragging = False

    elif event.endswith('_Selection'):
        selection_mode = event
        if event == "Block_Selection":
            # remake original rectangle collection
            page = doc[current_page]
            blocks = page.getText("dict")["blocks"]
            rectangle_collection = show_blocks_on_screen(blocks)
            blocks, rectangle_collection = initial_draw_and_process_page(page, blocks, statistics)

            for r in rectangle_collection:
                if r.type == Type.DISPLAY_EQUATION:
                    r.highlight(r.type)
        elif event == "Span_Selection": 
            # convert block_collection to span_collection
            span_collection = convert_to_span_collection(blocks, rectangle_collection, True) 

            # get rid of the highlighting on blocks
            for r in rectangle_collection:
                r.unhighlight(Type.TEXT)

            # replace rectangle_collection with a converted span collection
            rectangle_collection = span_collection

            for r in rectangle_collection:
                if r.type == Type.DISPLAY_EQUATION:
                    r.highlight(r.type)
        elif event == "Column_Selection":
            # remove all current columns from screen to make way for new ones
            statistics.remove_columns(page, graph)

    elif event.endswith('Mode'): # One of the old radio buttons was selected
        pass
        #  selection_mode = event

    elif event == "Prev": # Prev button was just pressed
        current_page -= 1
        # handle page turning
        page = doc[current_page]
        blocks = page.getText("dict")["blocks"]
        rectangle_collection = show_blocks_on_screen(blocks)
        blocks, rectangle_collection = initial_draw_and_process_page(page, blocks, statistics)
        window.Element('Block_Selection').Update(value=True)
        selection_mode ='Block_Selection' 

        for r in rectangle_collection:
            if r.type == Type.DISPLAY_EQUATION:
                r.highlight(r.type)
        window.Element("page_num").Update(value=f"{current_page}")

    elif event == "Next": # Next button was just pressed
        #  for use when pressing button
        current_page += 1
        # handle page turning
        page = doc[current_page]
        blocks = page.getText("dict")["blocks"]
        rectangle_collection = show_blocks_on_screen(blocks)
        blocks, rectangle_collection = initial_draw_and_process_page(page, blocks, statistics)

        # select block mode
        window.Element('Block_Selection').Update(value=True)
        selection_mode ='Block_Selection' 

        for r in rectangle_collection:
            if r.type == Type.DISPLAY_EQUATION:
                r.highlight(r.type)
        # TODO remake new page
        window.Element("page_num").Update(value=f"{current_page}")

    elif event == "Process":
        print_page_to_html(page, blocks, rectangle_collection, statistics, )
    else:
        print("unhandled event", event, values)



