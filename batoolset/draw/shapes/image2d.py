from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
from batoolset.lists.tools import bool_list_to_index_list
from batoolset.luts.lut_minimal_test import PaletteCreator, list_available_luts, apply_lut, lsm_LUT_to_numpy
from batoolset.files.tools import smart_name_parser
from batoolset.files.tools import get_consolidated_filename_from_parent
from batoolset.strings.tools import reload_string_list, is_string_a_list_string, find_letter_before_h
from batoolset.draw.shapes.automated_alignment_drawing_refactored import compute_translation_corrections
from batoolset.pyqt.tools import check_antialiasing, copy_transform_without_rotation, rotate_image, copy_scale_only, \
    copy_scale_and_translation_only, crop_image, get_shape_after_rotation_and_crop, tst_translation
from batoolset.serialization.tools import create_objects_from_dict, object_to_xml
from batoolset.matplolib.tools import get_fig_rect
import os
import traceback
from batoolset.draw.shapes.Position import groupby_position, Position
from batoolset.draw.shapes.rectangle2d import Rectangle2D
import numpy as np
import matplotlib.pyplot as plt
from qtpy import QtGui
from qtpy.QtGui import QPainter, QColor, QBrush, QPen, QTransform, QFontMetrics,QImage
from qtpy.QtCore import Qt
from qtpy.QtSvg import QSvgGenerator
from qtpy.QtSvg import QSvgRenderer
from batoolset.draw.shapes.line2d import Line2D
from batoolset.draw.shapes.point2d import Point2D
from batoolset.figure.alignment import alignRight, alignLeft, alignTop, alignBottom, alignCenterH, alignCenterV, packY, \
    packX, packYreverse, pack2, packing_modes
from batoolset.draw.shapes.scalebar import ScaleBar
from batoolset.draw.shapes.txt2d import TAText2D

from batoolset.img import Img, toQimage, RGB_to_BGR, guess_dimensions, max_proj, min_proj, avg_proj
from qtpy.QtCore import QRectF, QPointF, QSize, QRect
from qtpy.QtWidgets import QApplication
import sys
import io
from batoolset.tools.logger import TA_logger
import ast
import re

logger = TA_logger()


__DEBUG_NORMALIZATION__ = False

class Image2D(Rectangle2D):
    # min = None, max = None,
    def __init__(self, *args, x=None, y=None, width=None, height=None, data=None, dimensions=None, channels=None, LUTs=None, projection=None, normalization=None,  filename=None, isText=False, fraction_of_parent_image_width_if_image_is_inset=0.25, opacity=1., theta=None, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0,fill_color=None, border_color=0xFFFFFF,border_size=None, placement=Position('Top-Right'),annotations=None, custom_loading_script=None, allow_unknown_files=True, scale=1, user_comments='',__version__=1.0, px_to_unit_conversion_factor=0, extra_user_translation=(0,0),script_log=None, template_rect=None,**kwargs):

        # print('args',args)
        # if kwargs:
        #     print('ignored parameters', kwargs)

        if template_rect:
            try:
                # print('template_rect',template_rect)
                pattern = r'\((.*?)\)'
                match = re.search(pattern, template_rect)

                if match:
                    values_str = match.group(1)  # Get the matched substring
                    values = [float(x) for x in values_str.split(',')]  # Convert to floats

                    # Create a QRectF object
                    template_rect = QRectF(*values)
                    # print(template_rect)
            except:
                pass
        else:
            # print('no template rect')
            pass

        self.lower_indices = None
        self.upper_indices = None
        self.lower_values = None
        self.upper_values = None
        # maybe allow fix hot pixels error -−> TODO
        # self.min=min # will store the normalization min
        # self.max=max # will store the normalization max # nB normalization is mandatory for 16bits images and probably recommended for LUTs -> see how I should handle that
        self.projection = projection # to store if a max proj or alike should be made (I could in the future store first and last z for the projection)
        self.normalization = normalization # this will contain all the normalization infos as a dict
        self.isText = isText
        if isinstance(self.normalization, str):
            try:
                self.normalization = ast.literal_eval(self.normalization)
            except:
                pass
        self.channels = channels # will store the channel state (active or not)
        if isinstance(self.channels, str):
            try:
                self.channels = ast.literal_eval(self.channels)
            except:
                pass

        # print('inputed svchannels', self.channels)
        # print('inputed svchannelsb', channels)

        self.LUTs = LUTs # will store the user defined LUTs
        if isinstance(self.LUTs, str):
            try:
                self.LUTs = ast.literal_eval(self.LUTs)
            except:
                pass
        self.dimensions = dimensions
        if isinstance(self.dimensions, str):
            try:
                self.dimensions = ast.literal_eval(self.dimensions)
            except:
                pass




        self.script_log = script_log # a log of the screen so that the user can see and read the errors of their script -−> can be very useful for debug
        self.extra_user_translation = extra_user_translation
        self.px_to_unit_conversion_factor=px_to_unit_conversion_factor
        self.user_comments = user_comments # contains some useful hints for the person that made the figure
        # if this stuff is set then allow to load the image using that -−> TODO
        self.qimage = None
        self.custom_loading_script = custom_loading_script # TODO implement that --> should assign to self.img smthg that I can handle -−> either a raster or a svg or whatever # par exemple pr loader les certains frames d'une image -−> qq chose de facile à faire

        # crops
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.img = None
        self.line_style = None # useless yet just for compat
        if not annotations: # DEV NOTE KEEP WEIRD SERIALIZATION CODE IF I DONT DO THAT --> SO KEEP --> IF I INITITATE annotations = [] in the init then nested images in annotations cause an infinite loop in serialization, no clue why though
            annotations = []
        if isinstance(annotations, dict): # deprecated --> maybe remove that because it was successfully changed
            annotations=create_objects_from_dict(annotations)
        elif isinstance(annotations, list) and len(annotations)>0:
            if isinstance(annotations[0], dict):
                annotations = create_objects_from_dict(annotations)

        self.annotations=annotations
        if isinstance(placement, str):
            placement = Position(placement)
        self.placement=placement
        self.renderer = None # replaced all of these by self.img

        # if the image is inserted as an inset then draw it as a fraction of parent width
        # inset parameters
        self.fraction_of_parent_image_width_if_image_is_inset = fraction_of_parent_image_width_if_image_is_inset
        self.border_size = border_size  # no border by default
        self.border_color = border_color  # white border by default
        self.filename = filename

        if not args and filename:
            args = [filename]

        if args:
                if isinstance(args[0], str):
                    self.filename = args[0]
                elif isinstance(args[0], Img):
                    self.filename = None
                    self.img = args[0]
                    # self.qimage = toQimage(self.img)
                    self.update_qimage()
                    if x is None:
                        x = 0
                    if y is None:
                        y = 0

                    w,h = self.get_raw_size()
                    # try:
                    super(Image2D, self).__init__(x, y, w, h)
                    # except:
                    #     super(Image2D, self).__init__(x, y, self.img.shape[1], self.img.shape[0])
                elif isinstance(args[0], plt.Figure):
                    # the passed image is in fact a matplotlib plot --> init super with the corresponding QrectF
                    # super(Image2D, self).__init__(get_fig_rect(args[0]))
                    self.img = args[0]
                    w,h = self.get_raw_size()
                    super(Image2D, self).__init__(0,0,w,h)
                    self.setFigure(self.img)
                elif isinstance(args[0], np.ndarray):
                    # the only pb of that is that it cannot be reloaded -−> rather recommend the user to use scripts
                    self.img=args[0]
                    # self.qimage = toQimage(self.img)
                    self.update_qimage()

                    w,h = self.get_raw_size()
                    # try:
                    #     super(Image2D, self).__init__(0,0, self.img.get_width(), self.img.get_height())
                    super(Image2D, self).__init__(0,0, w, h)
                    # except:
                    #     super(Image2D, self).__init__(0,0, self.img.shape[1], self.img.shape[0])

        if x is None and y is None and width is not None and height is not None and self.filename is None:
            super(Image2D, self).__init__(0, 0, width, height)
            # self.img = QRectF(0, 0, width, height) # hack to fake an image... to allow crop and alike to always work
            # if not template_rect:
            #     self.img = QRectF(0, 0, width, height) # hack to fake an image... to allow crop and alike to always work
            # else:
            #     self.img = template_rect
        elif self.filename is not None: # and not isinstance(self.filename, TAText2D):
            # handle svg files first
            if self.filename.lower().endswith('.svg'):
                super(Image2D, self).__init__(0, 0, 512, 512) # fake definition
                self.setFigure(self.filename) #real loading of the parameters
            else:
                try:
                    self.img = Img(self.filename)
                    # self.qimage = toQimage(self.img)
                    self.update_qimage()
                    # width = self.qimage.width()
                    # height = self.qimage.height()

                    # print('width height defined here127', width, height, self.filename)
                    w,h = self.get_raw_size()
                    super(Image2D, self).__init__(0, 0, w, h)
                except:
                    if not self.custom_loading_script:
                        # if the file canot be opened and there is no script --> then there is chance that it works
                        if not self.filename == '@template@':
                            logger.error('could not load image '+str(self.filename))
                    if not allow_unknown_files:
                        return
                    else:
                        # define a default image with 512x512 and display error on top of it
                        # in fact this can be an empty image
                        self.img = None # this is the indication that an error occurred and that I cannot do much
                        super(Image2D, self).__init__(0, 0, 512, 512)
            if x is not None:
                self.setX(x)
            if y is not None:
                self.setY(y)
            if width is not None:
                self.setWidth(width)
            if height is not None:
                self.setHeight(height)
        elif x is not None and y is not None and width is not None and height is not None and self.img is None:
            self.img = None
            super(Image2D, self).__init__(x, y, width, height)
        elif self.filename is not None: # and not isinstance(self.filename, TAText2D):
                self.img = Img(self.filename)
                # self.qimage =  toQimage(self.img)
                self.update_qimage()
                if x is None:
                    x = 0
                if y is None:
                    y = 0
                w,h = self.get_raw_size()
                # super(Image2D, self).__init__(x, y, self.img.get_width(), self.img.get_height())
                super(Image2D, self).__init__(x, y, w,h)

        # try a ultimate rescue if there was a script
        if custom_loading_script:

            # why is image None there
            self.execute_code_unsafe(custom_loading_script) # MEGA TODO --> maybe I need to log the script error and put it somewhere in a widget -−>
            # print('entering custom loading script', self.img, self.script_log)
            if self.script_log:
                print('Script output:\n',self.script_log)
        if custom_loading_script and self.img is not None:
            # TODO --> avoid code duplication!!! --> TODO -> split my code into plenty of stuff
            if isinstance(self.img, plt.Figure):
                super(Image2D, self).__init__(get_fig_rect(self.img))
                self.setFigure(self.img)
            if isinstance(self.img, np.ndarray):
                # print('setting new stuff here')
                # self.qimage = toQimage(self.img)
                self.update_qimage()
                w, h = self.get_raw_size()
                # self.setWidth(self.qimage.width())
                # self.setHeight(self.qimage.height())
                # self.setWidth(w)
                # self.setHeight(h)
                super(Image2D, self).__init__(0,0, w, h)
        try:
            self.__init_called__
        except:
            if not allow_unknown_files:
                raise Exception('Image not initialized properly!, super-class __init__() was never called')
            else:
                super(Image2D, self).__init__(0, 0, 512, 512) # create an error image maybe the custom code can be edited later?
                # self.img=QRectF(0, 0, 512, 512)
                if not template_rect:
                    self.img = QRectF(0, 0, 512,
                                      512)  # hack to fake an image... to allow crop and alike to always work
                else:
                    self.img = template_rect

        # THIS SHOULD ALWAYS BE AT THE VERY END OTHERWISE SUPER OVERRIDES THE VALUES --> PROBABLY NEED TO DO IT IN A SMARTER WAY BUT OK FOR NOW !!!!
        self.opacity = 1
        self.scale = scale
        self.translation = QPointF()
        self.fill_color = fill_color  # can be used to define a bg color for rotated images (and or transparent images...)
        self.theta = theta


        if self.is_template(): # if image is converted to template then we must remove the conversion factor (maybe make this an option some day but I think it makes sense)
            self.px_to_unit_conversion_factor=0 # be sure to reset it if in a template

        if not self.px_to_unit_conversion_factor:
            # try get it from the image
            self.get_px_to_unit_conversion_factor_from_metadata()



        if template_rect:
            if self.img is None or isinstance(self.img, QRectF):
                self.img = template_rect
        self.__version__ = __version__

    def get_px_to_unit_conversion_factor_from_metadata(self):
        try:
            self.px_to_unit_conversion_factor = self.img.metadata['vx']
        except:
            self.px_to_unit_conversion_factor = 0
            # print()


    def get_extra_dimensions(self, handle_proj=False):
        if not isinstance(self.img, np.ndarray):
            return None
        guessed_dimensions = guess_dimensions(self.img)
        extra_dims = guessed_dimensions.replace('h', '').replace('w', '').replace('c', '')
        if extra_dims and handle_proj:
            if self.projection is not None:
                extra_dims = extra_dims[:-1]
        return extra_dims

    def clip_by_frequency(self, lower_cutoff=None, upper_cutoff=0.05, channel_mode=True):
        '''
        Clips the values of an image based on frequency cutoffs.

        Args:
            img (ndarray): Input image.
            lower_cutoff (float, optional): Lower frequency cutoff.
            upper_cutoff (float, optional): Upper frequency cutoff.
            channel_mode (bool, optional): Specifies whether to perform clipping on individual channels.

        Returns:
            ndarray: Clipped image.
        '''

        logger.debug(' inside clip ' + str(lower_cutoff) + str(upper_cutoff) + str(channel_mode))

        # Check if all cutoff values are 0 or None, in which case the image remains unchanged.
        if lower_cutoff == upper_cutoff == 0 or lower_cutoff == upper_cutoff == None:
            logger.debug('clip: keep image unchanged')
            return img

        # Check if either lower_cutoff or upper_cutoff is None while the other is 0, in which case the image remains unchanged.
        if (lower_cutoff is None and upper_cutoff == 0) or (upper_cutoff is None and lower_cutoff == 0):
            logger.debug('clip: keep image unchanged')
            return img

        logger.debug('chan mode ' + str(channel_mode))

        # If the maximum and minimum values of the image are the same, return the image as is.
        if img.max() == img.min():
            return img

        # If channel_mode is True, perform clipping on individual channels.
        if channel_mode:
            for ch in range(img.shape[-1]):
                img[..., ch] = self.clip_by_frequency(img[..., ch], lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff,
                                                 channel_mode=False)
            return img

        logger.debug('Removing image outliers/hot pixels')

        # Compute the maximum value for clipping based on the upper_cutoff.
        if upper_cutoff is not None:
            max = np.percentile(img, 100. * (1. - upper_cutoff))
            img[img > max] = max

        # Compute the minimum value for clipping based on the lower_cutoff.
        if lower_cutoff is not None:
            min = np.percentile(img, 100. * lower_cutoff)
            img[img < min] = min

        return img

    def update_qimage(self, reset_outlier_pos=False):
        # MEGA TODO NB SOME OF THE PARAMTERS PROBABLY NEED NOT BE CHANGED UNTIL IMAGE IS CHANGED --> SPLIT THIS IN INIT_IMAGE AND UPDATE_IMAGE AND CALL THE APPROPRIATE for example the bins/histo/min/max and so on need not be redone

        # TODO --> merge all so that I do not have to create yet another layer ??? -−> TODO
        # shall I use views to avoid memory overload !!!
        # TODO --> try to finalize all!!!

        # print('%%%'*32)
        # print('neo parameters', self.normalization, self.projection)
        # self.channels = 1

        # if reset_outlier_pos:
        #     self.remove_outliers()
        #     self.reset_outliers()

        # print('MEGA DEBUGGING',self.img.max())
        if __DEBUG_NORMALIZATION__:
            print('#'*20+'ENTERING UPDATE QIMAGE'+'#'*20)

        if not isinstance(self.img, np.ndarray):
            self.qimage = None
            # now I need to get the max and min per channel maybe if per channel is on
            return

        if True:
            # print('I am inside')
            per_channel = False
            if self.normalization is not None:
                if __DEBUG_NORMALIZATION__:
                    print('called')
                if self.normalization is not None:
                    if __DEBUG_NORMALIZATION__:
                        print('called 2')
                    if 'per_channel' in self.normalization:
                        per_channel = self.normalization['per_channel']

                if __DEBUG_NORMALIZATION__:
                    print('called 3')
                if 'ignore_hot_spots' in self.normalization and self.normalization['ignore_hot_spots'] is not None and  self.normalization['ignore_hot_spots'] is not False:
                    # print('TODO -−> NOW I NEED TO CHANGE THE NORMALIZATION -−> ACTION REQUIRED')
                    # # pass
                    if __DEBUG_NORMALIZATION__:
                        print('called 4')
                    # # self.reset_outliers()
                    # print('called 5')
                    #
                    # print('removing outliers')

                    self.remove_outliers(per_channel=per_channel)
                else:
                    if __DEBUG_NORMALIZATION__:
                        print('called 6')
                    pass
                    # self.restore_outliers()
                    # self.reset_outliers()
            else:
                if __DEBUG_NORMALIZATION__:
                    print('called 7')
                pass
                # self.restore_outliers()
                # self.reset_outliers()

        # print('MEGA DEBUGGING3', self.img.max())


        # maybe I could integrate in there the rotation (maybe also crop) and all alike stuff and max proj and also the
        image_to_display = self.img




        target_dims = guess_dimensions(self.img)
        # original_dims=target_dims

        # print('super init target_dims',target_dims)
        proj_letter = find_letter_before_h(target_dims)



        # parse the image if needed
        if isinstance(self.dimensions, dict):
            # print('checking order',self.dimensions['order'])
            # print('checking values',self.dimensions['values'])
            if len(self.dimensions['order'])!=len(self.img.shape) or target_dims!=self.dimensions['order']:
                print('dimension mismatch between dimensions and current image -−> ignoring')
            else:
                for ddd,dim_to_chose in enumerate(self.dimensions['values']):
                    if target_dims[ddd] == proj_letter and self.projection: # we keep the letter for projection if projection is selected
                        # if there is a dim dimension and the user wants a projection --> do not allow dimension reduction!!!
                        continue
                    if dim_to_chose>=0 and dim_to_chose<self.img.shape[ddd]:
                        image_to_display = image_to_display[dim_to_chose]
                    else:
                        # print('invalid dimension specified assuming image has changed and resetting the value')
                        image_to_display = image_to_display[0]

        if len(target_dims) > len(image_to_display.shape):
            target_dims = target_dims[len(target_dims)-len(image_to_display.shape):]

        # print('final rendering',image_to_display.shape)
        # print('checking img', image_to_display.shape, 'vs', self.img.shape)
        # if self.channels is not None and not self.channels == 'merge' and 'c' in target_dims:
        #         if self.channels>=0 and self.channels < self.img.shape[-1]:
        #             image_to_display=image_to_display[...,self.channels]
        #             # image_to_display=image_to_display[...,np.newaxis] # do this to avoid losing dimension
        #             target_dims = target_dims.replace('c','')

        # print('user defined Luts here', self.LUTs)

        # print('checking img', image_to_display.shape)
        # shall I allow LUTS for non channels images −−> if so add them a channel before that -−> but then need add a dimension

        if False:
            # try add a channel but somehow that creates a bug --> probably do not use that!!!
            if not 'c' in target_dims:
                image_to_display = image_to_display[..., np.newaxis]
                target_dims+='c'

        if self.channels and 'c' in target_dims:
            # print('inside channels sel self.channels', self.channels)
            list_of_true_indices = bool_list_to_index_list(self.channels)
            # print('list_of_true_indices',list_of_true_indices)
            if list_of_true_indices:
                image_to_display=image_to_display[...,list_of_true_indices] # that is funny the behaviour is different as a list keeps the last channel even if unique whererad the int removes it!!!!
            else:
                list_of_true_indices = [i for i,_ in enumerate(self.channels)] # if no index --> force select all for the LUTs --> make sens
            # print('list_of_true_indices fixed', list_of_true_indices)
            if image_to_display.shape[-1]==1:
                target_dims = target_dims.replace('c', '') # force it to be gray # so why is it colored this time ???
                image_to_display=image_to_display[...,0] # needed to remove the channel and the color # see how I can handle the LUts naow and do a serializable LUT object
                # print('removing c')


        # print('image_to_display after channel selection',image_to_display.shape)

        luts = None
        # if 'c'in original_dims:
        if True:
            if __DEBUG_NORMALIZATION__:
                print('entering Lut1')
                    # keep existing LUTs if they exist !!!
            try:
                if hasattr(self.img,'metadata') and 'LUTs' in self.img.metadata:
                    if self.img.metadata['LUTs']:
                        # print('checking luts',self.img.metadata['LUTs'])
                        # print('checking luts',len(self.img.metadata['LUTs']))
                        luts=self.img.metadata['LUTs']
                        # print(type(luts[0]))
                        # print('shape lut', luts[0].shape) # (3, 256)
                        # Use a list comprehension to select only the specified elements
                        if self.channels:
                            # luts = [luts[i] if channel_state else for i, channel_state in enumerate(self.channels)] # if there are LUTs they are applied --> I could also swap these LUts for the user chosen ones if any
                            # tmp = []
                            # for sss, state in enumerate(self.channels):
                            #     if state:
                            #         tmp.append(luts[sss])
                            # luts=tmp
                            tmp = [lut for lut, state in zip(luts, self.channels) if state] # we only get the luts for active states
                            if not tmp:
                                pass
                            else:
                                luts=tmp
                        else:
                            luts = luts
                            # I would need a LUT button and also to serialize the LUts --> maybe make an object

                        # palette = None
                        # if True:
                        #     # lut = self.lut_combo.currentText()
                        #     lutcreator = PaletteCreator()
                        #     luts_lst = lutcreator.list
                        #     # lut = lutcreator.create3(luts[lut])
                        #     try:
                        #         # palette = lutcreator.create3(luts[0])
                        #         palette = lutcreator.create3(luts_lst["RAINBOW_1"])# if LUT is from a string --> try save that, otherwise serialize the whole array in the sortest possible way
                        #         # palette = lutcreator.create3("GRAY") #
                        #     except:
                        #
                        #         # palette = None
                        #         # default to grey palette
                        #         palette = lutcreator.create3(luts_lst['GRAY'])
                        #
                        # print('shape lut', luts[0].shape)
                        # print('palette',palette)
                        # if palette is not None:
                        #     # ça marche je peux facilement remplacer une lut par ça en fait --> à faire -->
                        #     palette = np.asarray(palette)
                        #     print('shape pal', palette.shape)
                        #     luts[0]=palette
            except:
                traceback.print_exc()
                            # I need to update the dims ???

            # print('bobobob')
            # print('self.LUTs',self.LUTs, 'fixed LUTs', len(luts) if luts else luts) #--> ok
            # print('self.img.shape[-1]',self.img.shape[-1]) #--> ok

            final_luts_to_apply = []
            if luts:
                final_luts_to_apply.extend(luts)
            luts2 = []
            if self.LUTs and self.img.shape[-1]==len(self.LUTs): # TODO --> check that
                if __DEBUG_NORMALIZATION__:
                    print('entering Lut2')

                # keep only the LUTs of the specified user channels
                if self.channels:
                    # luts2 = [self.LUTs[i] for i, channel_state in enumerate(self.channels)]  # if there are LUTs they are applied --> I could also swap these LUts for the user chosen ones if any
                    # luts2 = [self.LUTs[i] for lut, state in zip(luts, self.channels) if state]  # we only get the luts for active states
                    luts2 = []
                    for i,state in enumerate(self.channels):
                        if state:
                            luts2.append(self.LUTs[i])
                    if not luts2: # nothing selected is the same as all is selected -−> fix all
                        luts2=self.LUTs
                else:
                    luts2 = self.LUTs


            if __DEBUG_NORMALIZATION__:
                print('checking luts2', luts2)
            # print('checking luts2', luts2) # vide
            # print('checking final_luts_to_apply', len(final_luts_to_apply))
            if luts2:
                if not final_luts_to_apply:
                    final_luts_to_apply.extend(luts2)
                else:
                    for l in range(len(final_luts_to_apply)):
                        final_luts_to_apply[l]=final_luts_to_apply[l] if luts2[l]==None else luts2[l]



            luts=final_luts_to_apply
            # print('###########')

        # no in fact max and min should be obtained there !!!!

        # print('checking img after', image_to_display.shape)
        #
        # print('target_dims',target_dims)
        #
        # print(len(target_dims), len(image_to_display.shape))

        # TODO --> hack qimage so that if more than 3 channels and blend I blend each channel and then  compute the sum or avg or max image of all the colors see how imageJ is doing and do the same!!!
        # check my LUTs and finalize that!!!
        self.qimage = None

        # print('dabouyo')

        # the real global max is there in fact -−> still I do not get it ???

        # print('megrabouya', self.channels)
        if __DEBUG_NORMALIZATION__:
            print('megrabouya', self.channels)

        # print('calling the prepare_image_for_qimage')
        try:
            self.prepare_image_for_qimage(image_to_display, target_dims, luts)
        except:
            traceback.print_exc()

        if self.qimage is None:
            # if __DEBUG_NORMALIZATION__:
            print('failed --> ROLLING BACK')
            # print('fixed target_dims', target_dims, image_to_display.shape)

            self.qimage = toQimage(image_to_display, normalize=False, dimensions=target_dims, luts=luts)
            # self.qimage = toQimage(image_to_display, dimensions=target_dims, luts=None)

        if __DEBUG_NORMALIZATION__:
            print('~'*30)

        # print('restoring and resetting outliers',self.img.max())
        self.restore_outliers()
        self.reset_outliers()
        # print(self.img.max())
        # restore is ok but display is not

        # print('~' * 30)

        del image_to_display
        # del self.img # that would be great but then I would need tricks to know its parameters

        # print('----------'*32)
        # return qimage

    def prepare_image_for_qimage(self, image_to_display, target_dims, luts, min=None, max=None, per_channel_normalization=False):
        # here I will prepare the image to be dispalyed as a qimage --> it can handle normalization, etc...
        # only keep from target_dims the hwc
        # -> flatten before by taking 0

        # we reduce the image until we have hwc left --> we stop at h unless we wanna do a max proj

        if __DEBUG_NORMALIZATION__:
            print('entering prepare_image_for_qimage')

        # for normalization -−> global (whole stack), selected or projected, or custom defined by the user to some min and some max values
        # --> TODO

        # if image_to_display is None:
        #     return

        list_of_true_indices = []
        try:
            list_of_true_indices = bool_list_to_index_list(self.channels)

            # print('guess_dimensions(self.img)',self.channels, guess_dimensions(self.img),list_of_true_indices)

            if not list_of_true_indices and 'c' in guess_dimensions(self.img):
                list_of_true_indices = [i for i, _ in enumerate(self.channels)]

            # print('list_of_true_indices', list_of_true_indices)
        except:
            pass


        # print('inside prepare_image_for_qimage')

        per_channel = False
        if self.normalization is not None:
            if 'per_channel' in self.normalization:
                per_channel = self.normalization['per_channel']

        global_max = self.img.max()
        global_min = self.img.min()


        # else:

        # default normalization -−> my favourite one definitely
        min = global_min
        max = global_max
        # I can also do it locally if there are several channels but then I need to keep only the channels ones!!!

        # print('shape of img for local', image_to_display.shape)



        # if self.projection:
        #     print('projection in prepare_image_for_qimage', self.projection)


        extra_dims =target_dims.replace('h','').replace('w','').replace('c','')

        # print('extra_dims',extra_dims)

        # the d version for normalization should be here
        # I haven't done the per channel but     this could be made later too

        final_target_dims = target_dims
        extra_dim_length = len(extra_dims)

        if extra_dims:
            if __DEBUG_NORMALIZATION__:
                print('extra_dims inside',extra_dims, image_to_display.shape)
            for zzz,extra in enumerate(extra_dims):
                if __DEBUG_NORMALIZATION__:
                    print('extra',extra)

                # print('check',zzz, extra, extra_dim_length, zzz==extra_dim_length-1)

                if self.projection and zzz==extra_dim_length-1:
                    if self.projection == 'max':
                        # print('running max proj on ', image_to_display.shape)
                        image_to_display = max_proj(image_to_display)
                        # print('running max proj on after ', image_to_display.shape)

                        # ça marche mais du coup pkoi pas de changement ????
                    elif self.projection == 'min':
                        image_to_display = min_proj(image_to_display)
                    else:
                        image_to_display = avg_proj(image_to_display)
                else:
                    image_to_display=image_to_display[0]
                final_target_dims=final_target_dims.replace(extra,'')

        local_max = image_to_display.max()
        local_min = image_to_display.min()

        if self.normalization is not None:
            if __DEBUG_NORMALIZATION__:
                if self.normalization:
                    print(self.normalization, self.normalization['range'], self.normalization['min_value'], self.normalization['max_value'])
            if 'range' in self.normalization and self.normalization['range'] == 'display':
                min = local_min
                max = local_max

            if 'range' in self.normalization and self.normalization['range'] == 'custom':
                min = self.normalization['min_value']
                max = self.normalization['max_value']



        if True:
            # we perform the normalization here!!!

            # print('image_to_display.max() before', image_to_display.max())
            # print('image_to_display.min() before', image_to_display.min())

            if __DEBUG_NORMALIZATION__:
                print('used_min 1', min)
            # print(image_to_display.shape)
                print('used_max 1', max)

            try:

                if __DEBUG_NORMALIZATION__:
                    print('main entrance', min, max)
                image_to_display = image_to_display.astype(float)

                # print('image_to_display.max() after', image_to_display.max())
                # print('image_to_display.min() after', image_to_display.min())



                if not per_channel and not 'c' in target_dims: # global
                    if __DEBUG_NORMALIZATION__:
                        print('NEXT STEP', min, max, self.normalization) # was ok and then I fuck it
                    # try:
                    #     print('NEXT STEP2a', self.normalization['range'])
                    # except:
                    #     pass
                    if 'c' in guess_dimensions(self.img) and list_of_true_indices:
                        # case of a single channel yet global max and min required but for a single channel
                        # case there is a single channel selected --> yet using global max
                        # for cur_ch in list_of_true_indices:

                        if False:
                            max = self.img[..., list_of_true_indices[0]].max()
                            min = self.img[..., list_of_true_indices[0]].min()
                        else:
                            # dirty fix for display overwriting error
                            if not self.normalization or self.normalization['range']=='all':
                                max = self.img.max()
                                min = self.img.min()
                    else:
                        max = self.img.max()
                        min = self.img.min()

                    if __DEBUG_NORMALIZATION__:
                        print('NEXT STEP2', min, max)

                    if self.normalization is not None:
                        # print('entering')
                        if 'range' in self.normalization and self.normalization['range'] == 'custom':
                            min = self.normalization['min_value']
                            max = self.normalization['max_value']
                        if 'range' in self.normalization and self.normalization['range'] == 'all':
                            min = self.img.min()
                            max = self.img.max()
                    else:
                        min = self.img.min()
                        max = self.img.max()

                    if __DEBUG_NORMALIZATION__:
                        print('NEXT STEP3', min, max)
                    #
                    # print('selected max min non ch mode', self.img.shape, min, max, self, list_of_true_indices, target_dims, per_channel, not per_channel or not 'c' in target_dims)

                    if max == min:
                        # image_to_display = ((image_to_display - min) / (max - min + 1e-9)) * 255.  # we just avoid div by 0
                        # if max !=0:
                            image_to_display.fill(0)
                        # else:
                        #     image_to_display = image_to_display.fill(max)
                    else:
                        image_to_display = ((image_to_display - min) / (max - min)) * 255.  # we just avoid div by 0
                        # to speed up the process use the other code
                else:
                    if __DEBUG_NORMALIZATION__:
                        print('per_channel', min, max, list_of_true_indices, target_dims) # too many dims --> error
                    # print('list_of_true_indices',list_of_true_indices, self.channels, self.img.shape[-1],image_to_display.shape[-1], target_dims)

                    if not list_of_true_indices and 'c' in target_dims:
                        list_of_true_indices = [i for i in range(self.img.shape[-1])]

                    if __DEBUG_NORMALIZATION__:
                        print('vale before -1', min, max, list_of_true_indices)
                    if list_of_true_indices:
                        for ccc,ch in enumerate(list_of_true_indices):
                            if per_channel and self.normalization and self.normalization['range'] == 'display' and 'c' in target_dims:
                                max = image_to_display[...,ccc].max()
                                min = image_to_display[...,ccc].min()
                            if __DEBUG_NORMALIZATION__:
                                print('vale before 0', min, max)
                            if per_channel and ((self.normalization and self.normalization['range'] == 'all') or not self.normalization):
                                max = self.img[...,ch].max()
                                min = self.img[...,ch].min()

                            # bug is here
                            if __DEBUG_NORMALIZATION__:
                                print('vale before 1',min, max )

                            if self.normalization is not None:
                                if 'range' in self.normalization and self.normalization['range'] == 'custom':
                                    min = self.normalization['min_value']
                                    max = self.normalization['max_value']

                            if __DEBUG_NORMALIZATION__:
                                print('selected max min ch', ccc, 'vs',ch, self.img.shape,image_to_display.shape,  min, max, self,list_of_true_indices, target_dims)
                            if max == min:
                                # image_to_display = ((image_to_display - min) / (max - min + 1e-9)) * 255.  # we just avoid div by 0
                                # if max !=0:
                                if 'c' in target_dims:
                                    image_to_display[...,ccc].fill(max)
                                else:
                                    image_to_display.fill(0)
                                # else:
                                #     image_to_display = image_to_display.fill(max)
                            else:
                                if 'c' in target_dims:
                                    image_to_display[...,ccc] = ((image_to_display[...,ccc] - min) / (max - min)) * 255.  # we just avoid div by 0
                                else:
                                    image_to_display = ((image_to_display - min) / (max - min)) * 255.  # we just avoid div by 0
                                # to speed up the process use the other code
                    else:
                        if __DEBUG_NORMALIZATION__:
                            print('no channel found --> normalization done independently')
                        image_to_display = ((image_to_display - min) / (max - min)) * 255.

                if __DEBUG_NORMALIZATION__:
                    print('image_to_display.max() after2', image_to_display.max())
                    print('image_to_display.min() after2', image_to_display.min())

                # print('image_to_display min max', image_to_display.min(), image_to_display.max())

                np.clip(image_to_display, 0, 255, out=image_to_display) # do I need that probably only if the value iiiis less than global max
                image_to_display = image_to_display.astype(np.uint8) # do I need that ???

                if __DEBUG_NORMALIZATION__:
                    print('image_to_display.max() after3', image_to_display.max())
                    print('image_to_display.min() after3', image_to_display.min())
            except:
                traceback.print_exc()

            if __DEBUG_NORMALIZATION__:
                print('max-min exit', max - min)

        # can do the same with the per_channel version −−> for which I should loop


        # print('final_target_dims', final_target_dims, 'target_dims',target_dims)

        # in fact if any projection needs be done it should be done here
        # maybe I could also exclude some hot pixels in there --> see how I can do that

        # target dim should be hw or hwc!!!
        # try to apply LUT maybe here

        # maybe here --> if no L

        if luts:
            if __DEBUG_NORMALIZATION__:
                print('ENTERING LUTS', min, max)
            # print('oubsi')
            available_luts = list_available_luts()
            for iii, lut in enumerate(luts):
                if not isinstance(lut, np.ndarray):
                    # print('palette')

                # else:
                    # print(lut)
                    lutcreator = PaletteCreator()
                    lut_list = lutcreator.list
                    # lut = lutcreator.create3(luts[lut])

                    # if isinstance(lut, list):
                    #     print('to be fixed',lut) # [255, 0, 0, 0] --> convert this to a lut --> easy

                    if isinstance(lut, str):
                        if lut in available_luts:
                            try:
                                luts[iii] = np.asarray(lutcreator.create3(lut_list[lut]))
                            except:
                                traceback.print_exc()
                                luts[iii] = None
                    elif isinstance(lut, list) and len(lut)==4:
                        luts[iii]=lsm_LUT_to_numpy(lut)
        else:
            if __DEBUG_NORMALIZATION__:
                print('FAILEUR NO LUT CASE', min, max)
            pass


        merge = None
        if 'c' in target_dims:
            if __DEBUG_NORMALIZATION__:
                print('ENTERING LAST C LOOP', min, max)
            # print('oubsi2')

            list_of_true_indices = []
            try:
                list_of_true_indices = bool_list_to_index_list(self.channels)
                # print('list_of_true_indices', list_of_true_indices)
                if not list_of_true_indices:
                    list_of_true_indices = [i for i, _ in enumerate(self.channels)]
            except:
                pass
                # apply LUT if there is a LUT

            # why do I loop this 4 times whereas I just want one ???
            for ch in range(image_to_display.shape[-1]):
                if __DEBUG_NORMALIZATION__:
                    print('channel', ch, image_to_display.shape)

                img = image_to_display[...,ch]

                # print('img slice shape', img.shape)





                # local_min = img.min()
                # local_max = img.max()




                # we normalize the images always in the same way and as defined by the user
                # if max!=min:
                # img = ((img-min)/(max-min+1e-9))*255. # we just avoid div by 0
                # img = np.clip(img, 0, 255).astype(np.uint8)
                # I can also use the global if wanted

                # I need to get the mode for the normalization


                # print('checks',luts, image_to_display.shape[-1]==len(luts), image_to_display.shape[-1],len(luts))



                if luts is not None and image_to_display.shape[-1]==len(luts) and luts[ch] is not None:

                    if __DEBUG_NORMALIZATION__:
                        print('entering LUT finalization')
                    # print('oubsi3')
                    # img = ((img-min)/(max-min))*255.
                    # img = np.clip(img, 0,255) # force all stuff that need a lut clip to something like that
                    # img = img.astypr(np.uint8)
                    #
                    # now I apply the LUT

                    # print('ch inside', ch, len(luts), luts[ch].shape)

                    # img = apply_lut(img, luts[ch], convert_to_RGB=True, min=min, max=max)
                    img = apply_lut(img, luts[ch],  convert_to_RGB=True, do_not_interpolate=True) # is it renormalizing here --> maybe I should prevent it

                    # img=luts[ch][img]


                    # print('oubsi3-2', img.shape)

                    # plt.close('all')
                    # plt.imshow(img)
                    # plt.show()



                else:

                    if __DEBUG_NORMALIZATION__:
                        print('SKIPPING LUT')
                    # print('dabouyo')
                    # print('img.shape', img.shape)
                    # print('*img.shape',*img.shape)
                    # print('(*img.shape, 3)',(*img.shape, 3))

                    # I still need to normalize the image in there !!!

                    try:
                        real_ch = list_of_true_indices[ch]
                    except:
                        real_ch=ch
                    tmp = np.zeros(shape=(*img.shape, 3), dtype=np.uint8)

                    if __DEBUG_NORMALIZATION__:
                        print('dabouyo2')
                    if real_ch < tmp.shape[-1]:
                        tmp[..., real_ch] = img
                    else:
                        # print('too many channels to make any sense anyway --> setting to white')
                        tmp[..., 0] = img
                        tmp[..., 1] = img
                        tmp[..., 2] = img
                    # otherwise I would need to apply LUTs in a specified order -−> and maybe loop
                    # or worst case scenario --> do it white

                    # print('dabouyo3')
                    img = tmp.astype(np.uint8)
                    # print('dabouyo4')

                    # print('dabouyo end of shape', img.shape)


                    # I also need to normalize the image like the others --> see how to do that




                # print('snip')
                if merge is None:
                    # print('in there')
                    merge = img
                    # print('merge after init', merge.shape)
                else:
                    # print('in there2')


                    # if not merge.dtype == i

                    merge = np.maximum(merge, img)
                # print('snap')
                # print('merge end of loop', merge.shape)

            # print(merge)


            # WHY THE FUCK IS THIS NEVER REACHED !!!!
            # print('snip2')
            # if merge is not None:
            #     print('inside da stuff', merge.shape)
            # else:
            #     print('not merged')
            # print('snap2')



            self.qimage = toQimage(merge, normalize=False,dimensions=final_target_dims, luts=None)
            # self.qimage = toQimage(merge, dimensions=None, luts=None)

            if __DEBUG_NORMALIZATION__:
                print('SUCCESS!!!')
        # else:
        #
        #     self.qimage = toQimage(image_to_display, normalize=False, dimensions=final_target_dims, luts=None)

            # if lut needs be applied and image is not normalized then need normalize it
        else:
            if __DEBUG_NORMALIZATION__:
                print('ENTERING LAST C LOOP ALTERNATIVE', min, max)
            # I still need to apply the lut if it exists
            # no channels -−> we can still apply the LUT
            if self.LUTs:
                if __DEBUG_NORMALIZATION__:
                    print('ENTERING LAST C LOOP ALTERNATIVE part1', min, max, len(self.LUTs))
                # code duplication --> shall I really do that ??? --> maybe have a force gray that makes it gray instead -− >better
                luts=[]
                available_luts = list_available_luts()
                for iii, lut in enumerate(self.LUTs):
                    if not isinstance(lut, np.ndarray):
                        # print('palette')
                    # else:
                        # print(lut)
                        lutcreator = PaletteCreator()
                        lut_list = lutcreator.list
                        # lut = lutcreator.create3(luts[lut])
                        if lut in available_luts:
                            try:
                                luts.append(np.asarray(lutcreator.create3(lut_list[lut])))
                            except:
                                traceback.print_exc()
                                # luts[iii] = None
                                luts.append(None)
                        else:
                            luts.append(None)
                    else:
                        luts.append(lut)
                if luts:

                    if __DEBUG_NORMALIZATION__:
                        print('THERE IS A LUT',len(luts)) # there is a bug in the the right lut is not selected
                    list_of_true_indices = []
                    try:
                        list_of_true_indices = bool_list_to_index_list(self.channels)
                        # print('list_of_true_indices', list_of_true_indices)
                        if not list_of_true_indices:
                            list_of_true_indices = [i for i, _ in enumerate(self.channels)]
                    except:
                        pass


                    # image_to_display=luts[0][image_to_display]
                    # just need the mapped lut else lut[0]
                    # try:
                    #     image_to_display = apply_lut(image_to_display,luts[list_of_true_indices[0]], convert_to_RGB=True)
                    # except:
                        # I need really get the right index

                    # print('I am called with index 0', self.channels, list_of_true_indices)

                    # there is a bug there if the image has multichannels I need get the real index of the selected channel --> this will not be 0
                    if list_of_true_indices: # seems to be required for proper mapping of LUTs for single channel images but maybe check


                        # print('mode1')
                        try:
                            if __DEBUG_NORMALIZATION__:
                                print('mode2', list_of_true_indices, len(luts)) # there are two luts --> see
                            image_to_display = apply_lut(image_to_display, luts[list_of_true_indices[0]],
                                                     convert_to_RGB=True, do_not_interpolate=True)
                        except:
                            if __DEBUG_NORMALIZATION__:
                                print('mode3')
                            # try:
                            #     image_to_display = apply_lut(image_to_display, luts[0],
                            #                                  convert_to_RGB=True, do_not_interpolate=True)
                            # except:
                            #     print('mode3b')
                            #     pass
                            pass
                    else:
                        if __DEBUG_NORMALIZATION__:
                            print('mode4')
                        image_to_display = apply_lut(image_to_display, luts[0],
                                                     convert_to_RGB=True, do_not_interpolate=True)

                # img = lut[ch][img]
            # print('inside da stuff2', image_to_display.shape)
            # self.qimage = toQimage(image_to_display, dimensions=final_target_dims,luts=luts)

            # print('passing there', image_to_display.shape,self.qimage , min, max)
            # so this means

            self.qimage = toQimage(image_to_display, normalize=False,dimensions=final_target_dims,luts=None)

        if __DEBUG_NORMALIZATION__:
            print('out final', image_to_display.shape, image_to_display.max(),image_to_display.min(), min, max, self.qimage) # min and max are incorrect too
        # else just return stuff !!!!
        del image_to_display

    def remove_outliers(self, lower_cutoff=None, upper_cutoff=0.001, per_channel=False):

        try:
            # print('#-'*32)
            self.restore_outliers()

            if not 'c' in guess_dimensions(self.img):
                per_channel=False

            if per_channel and isinstance(self.img, np.ndarray) and 'c' in guess_dimensions(self.img):
                self.lower_indices=[]
                self.upper_indices=[]
                self.lower_values=[]
                self.upper_values=[]

                if not lower_cutoff and not upper_cutoff:
                    return

                if self.img.max() == self.img.min():
                    return

                for ch in range(self.img.shape[-1]):
                    one_channel_img = self.img[...,ch]

                    if one_channel_img.max()==one_channel_img.min():
                        self.lower_indices.append(None)
                        self.upper_indices.append(None)
                        self.lower_values.append(None)
                        self.upper_values.append(None)
                        continue

                    if upper_cutoff:
                        # Find indices of values above upper_threshold
                        max = np.percentile(one_channel_img, 100. * (1. - upper_cutoff))
                        self.upper_indices.append(np.where(one_channel_img > max))
                        # Store original values
                        self.upper_values.append(one_channel_img[self.upper_indices[-1]])
                        # Replace outliers with placeholders
                        one_channel_img[self.upper_indices[-1]] = max

                    if lower_cutoff:
                        # Find indices of values below lower_threshold
                        min = np.percentile(img, 100. * lower_cutoff)
                        self.lower_indices.append(np.where(one_channel_img < min))
                        # Store original values
                        self.lower_values.append(one_channel_img[self.lower_indices[-1]])
                        # Replace outliers with placeholders
                        one_channel_img[self.lower_indices[-1]] = min

                    self.img[..., ch]=one_channel_img
            else:
                self.lower_indices = None
                self.upper_indices = None
                self.lower_values = None
                self.upper_values = None

                if not lower_cutoff and not upper_cutoff:
                    return

                if self.img.max() == self.img.min():
                    return

                if upper_cutoff:
                    # Find indices of values above upper_threshold
                    max = np.percentile(self.img, 100. * (1. - upper_cutoff))
                    self.upper_indices = np.where(self.img > max)
                    # Store original values
                    self.upper_values = self.img[self.upper_indices]
                    # Replace outliers with placeholders
                    self.img[self.upper_indices] = max

                    # print('gripsous', max)


                if lower_cutoff:
                    # Find indices of values below lower_threshold
                    min = np.percentile(img, 100. * lower_cutoff)
                    self.lower_indices = np.where(self.img < min)
                    # Store original values
                    self.lower_values = self.img[self.lower_indices]
                    # Replace outliers with placeholders
                    self.img[self.lower_indices] = min

            # print('-+' * 32)
        except:
            pass


        # return self.lower_indices, self.lower_values, self.upper_indices, self.upper_values

    def restore_outliers(self):
        try:
            # print('in restore_outliers ', self.img.max())
            # Restore original values
            if self.lower_indices is not None:
                if isinstance(self.lower_indices, list):
                    for iii, indices in enumerate(self.lower_indices):
                        if indices is not None:
                            # self.img[indices,...,iii] = self.lower_values[iii]
                            tmp = self.img[..., iii]
                            tmp[indices] = self.lower_values[iii]
                            self.img[..., iii] = tmp
                else:
                    self.img[self.lower_indices] = self.lower_values
            if self.upper_indices is not None:
                # print('in restore_outliers 2')


                if isinstance(self.upper_indices, list):
                    # print('in restore_outliers 3')
                    for iii,indices in enumerate(self.upper_indices):
                        if indices is not None:
                            tmp = self.img[...,iii]
                            tmp[indices] = self.upper_values[iii]
                            self.img[..., iii]=tmp
                else:
                    # print('in restore_outliers 4')
                    # print('beforere',self.upper_values.max(), self.img.max())

                    self.img[self.upper_indices] = self.upper_values
                    # print('afeter', self.upper_values.max(), self.img.max())
            else:
                # print('in restore_outliers NONE FOUND')
                pass
            # print('outside restore_outlier', self.img.max())
        except:
            pass

    def reset_outliers(self):
        try:
            # print('reset_outliers called')
            try:
                del self.lower_indices
                del self.lower_values
            except:
                pass

            self.lower_indices = None
            self.lower_values = None

            try:
                del self.upper_indices
                del self.upper_values
            except:
                pass

            self.upper_indices = None
            self.upper_values = None
        except:
            pass

    # centralized code for
    def get_raw_size(self):
        if self.img is not None and isinstance(self.img, np.ndarray):
            try:
                w = self.img.get_width()
                h = self.img.get_height()
                # print('hw is defined here 1', h,w, self.filename)
                return w,h
            except:
                pass

            if len(self.img.shape) <= 3:
                w = self.img.shape[1]
                h = self.img.shape[0]
            # elif len(self.img.shape) == 4:
            #     w = self.img.shape[2]
            #     h = self.img.shape[1]
            else: # assume hwc or more
                w=self.img.shape[-2]
                h=self.img.shape[-3]

            # print('hw is defined here 2', h, w, self.filename)
            return w,h
        elif isinstance(self.img, plt.Figure):
            rect = get_fig_rect(self.img)
            w = rect.width()
            h = rect.height()
            # print('hw is defined here 3', h, w, self.filename)
            return w,h
        elif isinstance(self.renderer, QSvgRenderer):
            # if figure is a svg file then get it from here
            w,h = self.renderer.defaultSize().width(),self.renderer.defaultSize().height()
            # print('hw is defined here 4', h, w, self.filename)
            return w,h

        if isinstance(self.img, QRectF):
            return int(self.img.width()), int(self.img.height())
        # this is problematic and should never be reached!!!

        # print('data to check',self.img, type(self.img), isinstance(self.img, QRectF), isinstance(self.img, str))

        # raise Exception('do nbot reach that to avoid issues')

        # if all failed just take the rect default size
        w = super().width()
        h = super().height()
        # print('hw is defined here 5', h, w, self.filename)
        return w,h


    def get_consolidated_filename(self, parent_file):
        self.filename = get_consolidated_filename_from_parent(self.filename, parent_file)


    def consolidate_names(self, files_to_consolidate, consolidated_files):
        if not self.filename:
            return
        if not  is_string_a_list_string(self.filename):
            idx = files_to_consolidate.index(self.filename)
            if idx!=-1:
            # if self.filename in files_to_consolidate:
                self.filename = consolidated_files[idx]
        else:
            files = reload_string_list(self.filename)
            consolidated_file_names = []
            for file in files:
                idx = files_to_consolidate.index(file)
                if idx != -1:
                    # if self.filename in files_to_consolidate:
                    consolidated_file_names.append(consolidated_files[idx])
            self.filename = str(consolidated_file_names)



    def _toBuffer(self, bufferType='raster'):
        if self.img is not None:
            buf = io.BytesIO()
            if bufferType == 'raster':
                self.img.savefig(buf, format='png', bbox_inches='tight')
            else:
                self.img.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            return buf
        return None

    def get_extra_border_size(self):
        if self.border_size is not None and self.border_color is not None:
            return self.border_size
        return 0

    def is_template(self):
        return self.filename == '@template@'

    def set_rotation(self, theta): # by default set it to black or to None --> white could also be useful
        super().set_rotation(theta)
        self.theta = theta # CHECK WHY AND IF I NEED THE LINE ABOVE!!!

    def draw_bg(self, painter, draw=True, parent=None, restore_painter_in_the_end=True):

        # drawing was getting too big so I split the draw bg and draw annotations into two separate things
        if draw:
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.NoBrush)

            if parent is None:
                painter.setClipRect(self.getRect(all=True))  # required first to fraw the stuff at the very beginning of it !!! --> it must be done before save

            if self.fill_color is not None:
                painter.setBrush(QBrush(QColor(self.fill_color)))
                painter.drawRect(self.getRect(all=True))

            try:  # trick to force equal save and restore even upon error !!!
                painter.setOpacity(self.opacity)

                if parent is not None:
                    rect_to_plot = QRectF(0, 0, parent.width(all=True) * self.fraction_of_parent_image_width_if_image_is_inset, 0)
                    rect_to_plot.setHeight(rect_to_plot.width() / (self.getRect().width() / self.getRect().height()))

                    self.set_to_scale(self.width() / rect_to_plot.width())  # TODO --> CHECK THAT -->  is that ok or should I do the opposite ??? --> this is the only element that needs be rescaled because all the other shold not be changed
                    rect_to_plot = rect_to_plot.translated(self.topLeft())  # very hacky way to get it to work --> see how I can do that #  DEV NOT THIS IS MANDATORY TO GET THE INSETS TO DISPLAY --> NEVER REMOVE THIS
                else:
                    rect_to_plot = self.getRect(all=True)  # in that case I can add it a position

                if self.img is not None and not isinstance(self.img, QRectF) and not isinstance(self.img, plt.Figure) and not isinstance(self.img, QSvgRenderer):
                    x = 0
                    y = 0
                    # try:
                    #     w = self.img.get_width()
                    #     h = self.img.get_height()
                    # except:
                    #     # see how to handle images that are not in the stuff
                    #     # TODO maybe fix that too
                    #     w = self.img.shape[1]
                    #     h = self.img.shape[0]
                    #
                    # if w is None or h is None:
                    #     w = self.img.shape[1]
                    #     h = self.img.shape[0]
                    w,h = self.get_raw_size()

                    if not self.theta:
                        if self.crop_top is not None:
                            y = self.crop_top
                            h -= self.crop_top
                        if self.crop_left is not None:
                            x = self.crop_left
                            w -= self.crop_left
                        if self.crop_right is not None:
                            w -= self.crop_right
                        if self.crop_bottom is not None:
                            h -= self.crop_bottom

                    # pb here --> see how to really crop
                    qsource = QRectF(x, y, w, h)

                    # TODO --> maybe have a look here some day https://stackoverflow.com/questions/15166754/stroking-a-path-only-inside-outside

                    if parent is not None and self.border_color is not None and self.border_size is not None and self.border_size >= 1:  # we draw the square first then the image so that the image is not truncated by the bounding rect
                        pen = QPen(QColor(self.border_color),
                                   self.border_size * 2.)  # Adjust the color and line style as needed
                        painter.setPen(pen)
                        painter.drawRect(rect_to_plot)

                    if not self.theta:
                    # original code -−> was not working if image was rotated --> needed a fix
                        painter.drawImage(rect_to_plot, self.qimage, qsource)  # , flags=QtCore.Qt.AutoColor
                    else:
                        # VERY DIRTY HACK --> DRAW ALL THE NON PLACED SHAPES ON THE ORIGINAL IMAGE THEN ROTATE IT -−> MEGA DIRTY HACK BUT WILL WORK !!!
                        x = 0
                        y = 0
                        # try:
                        #     w = self.img.get_width()
                        #     h = self.img.get_height()
                        # except:
                        #     # see how to handle images that are not in the stuff
                        #     w = self.img.shape[1]
                        #     h = self.img.shape[0]
                        #
                        # if w is None or h is None:
                        #     w = self.img.shape[1]
                        #     h = self.img.shape[0]

                        w,h=self.get_raw_size()

                        image_rect = QRectF(0, 0, w, h) # maybe w+1 and h+1 is more correct --> need think and check
                        center_img = image_rect.center()

                        init_rect = QRectF(image_rect.x(), image_rect.y(), image_rect.width(), image_rect.height())
                        transformation_for_image=QTransform()
                        transformation_for_image.translate(center_img.x(), center_img.y())
                        transformation_for_image.rotate(self.theta)
                        transformation_for_image.translate(-center_img.x(), -center_img.y())

                        # we center the rotated rect to that
                        rotated_image_rect = transformation_for_image.mapRect(init_rect)

                        rotated_image_rect.translate(center_img.x()-rotated_image_rect.center().x(), center_img.y()-rotated_image_rect.center().y())

                        if self.crop_top is not None:
                            rotated_image_rect.setY(rotated_image_rect.y() + self.crop_top)  # but the crop should be much less of the real image if
                        if self.crop_left is not None:
                            rotated_image_rect.setX(rotated_image_rect.x() + self.crop_left)
                        if self.crop_right is not None:
                            rotated_image_rect.setWidth(rotated_image_rect.width() - self.crop_right)
                        if self.crop_bottom is not None:
                            rotated_image_rect.setHeight(rotated_image_rect.height() - self.crop_bottom)

                        rotated_image = rotate_image(self.qimage, self.theta, annotations=None,  parent_painter=painter) # DEV NOT KEEP there is a bug BE CAREFUL CALLING A QRECTF(ANOTHER) WITH ANOTHER that has negative x sets it to 0 and reduces its width and height --> NO CLUE WHY !!!!
                        cropped_rotated_image = crop_image(rotated_image, self.crop_left, self.crop_right, self.crop_top, self.crop_bottom)

                        rotated_image_rect = QRectF(0, 0, cropped_rotated_image.width(), cropped_rotated_image.height())
                        rotated_image_target = QRectF(rotated_image_rect.x(), rotated_image_rect.y(), rotated_image_rect.width(), rotated_image_rect.height())
                        rotated_image_target.setWidth(rotated_image_target.width()/self.scale)
                        rotated_image_target.setHeight(rotated_image_target.height()/self.scale)

                        rotated_image_target.translate(rect_to_plot.center()-rotated_image_target.center())
                        painter.drawImage(rotated_image_target, cropped_rotated_image, rotated_image_rect)

                elif self.renderer is not None or isinstance(self.img, plt.Figure):
                    if self.crop_left or self.crop_right or self.crop_top or self.crop_bottom or self.theta:
                    # if True:
                        # almost all ok but somehow the new rendered does introduce a small size difference that I don't get and that was not in previous if no rotation -−> see

                        # Get the viewBox of the SVG file
                        # view_box = self.renderer.viewBoxF()
                        # Print the viewBox information
                        # print(f"ViewBox: {view_box}", self.renderer.defaultSize(), self.getRect(all=True), rect_to_plot)
                        # the pb is size is already int rounded --> may cause errors --> see if I can do something better


                        # New code with crop and rotations for svg in the same way as for figs
                        painter.save()
                        rect_to_plot = self.getRect(all=True)  # yes indeed it should be cropped -−> ok
                        w,h = self.get_raw_size()
                        svg_rect = QRectF(0, 0, w,h)

                        center = svg_rect.center()

                        # if self.theta:
                        rotation_transform = QTransform().translate(center.x(), center.y()).rotate(self.theta if self.theta else 0).translate(-center.x(), -center.y()) # MEGA TODO --> replace by Transform2D, just make a version that supports maprect and maptopoly to make it easy to use

                        # Calculate the bounding rectangle of the rotated SVG
                        bounding_rect = rotation_transform.mapRect(svg_rect)
                        # else:
                        #     bounding_rect = QRectF(svg_rect)

                        # Crop the bounding rectangle
                        cropped_bounding_rect = bounding_rect.adjusted(self.crop_left, self.crop_top, -self.crop_right, -self.crop_bottom)
                        # cropped_bounding_rect = QRectF(bounding_rect.x()-self.crop_left, bounding_rect.y()-self.crop_top, bounding_rect.width()-self.crop_left-self.crop_right, bounding_rect.height()-self.crop_top-self.crop_bottom)

                        self.clip_rect = rect_to_plot

                        # Calculate the scale factor to fit cropped_bounding_rect into self.clip_rect
                        scale_factor = min(self.clip_rect.width() / cropped_bounding_rect.width(), # this one generates some small borders --> None of these is perfect (where is the AR error comming from, rounding errors ???? ???
                                           self.clip_rect.height() / cropped_bounding_rect.height())
                        if False:
                            scale_factor = max(self.clip_rect.width() / cropped_bounding_rect.width(), # this one is a bit out
                                           self.clip_rect.height() / cropped_bounding_rect.height())
                            scale_factor = (self.clip_rect.width() / cropped_bounding_rect.width()+  #
                                           self.clip_rect.height() / cropped_bounding_rect.height())/2
                        #
                        scale_factors = self.clip_rect.width() / cropped_bounding_rect.width(), self.clip_rect.height() / cropped_bounding_rect.height()

                        # is there an error or scale both ???


                    # Scale and translate the painter to align cropped_bounding_rect with self.clip_rect
                        scaled_cropped_bounding_rect = QRectF(cropped_bounding_rect.x()* scale_factor, cropped_bounding_rect.y()* scale_factor, cropped_bounding_rect.width()* scale_factor, cropped_bounding_rect.height() * scale_factor)
                        painter.translate(self.clip_rect.center() - scaled_cropped_bounding_rect.center())

                        # WHY ARE ARs not the same ????
                        # print('comparing ARs', self.clip_rect.width() / cropped_bounding_rect.width(), self.clip_rect.height() / cropped_bounding_rect.height(), self.clip_rect.height() / cropped_bounding_rect.height()==self.clip_rect.width() / cropped_bounding_rect.width())

                        # Scale the painter
                        if True:
                            painter.scale(scale_factor, scale_factor)
                        else:
                            painter.scale(scale_factors[0], scale_factors[1])
                            # painter.scale(scale_factors[1], scale_factors[0])

                        if self.theta:
                            painter.translate(svg_rect.center())
                            painter.rotate(self.theta if self.theta else 0)
                            painter.translate(-svg_rect.center())

                        # is this code changing AR somehow ???

                        # this does not work but that should --> so why is that
                        # svg_rect2 = QRectF(0,0, svg_rect.width()*scale_factors[0], svg_rect.height()*scale_factors[1])
                        # svg_rect2.translate(svg_rect.center()-svg_rect2.center())

                        self.renderer.render(painter, svg_rect)
                        # self.renderer.render(painter, svg_rect2)

                        painter.restore()
                    else:
                        painter.save()
                        if self.theta:  # restored rotation of svgs
                            # # I have added support for rotation to that maybe I will need something smarter if rotation 90 degrees or flips and images no same width and height
                            painter.translate(rect_to_plot.center())
                            painter.rotate(self.theta)
                            painter.translate(-rect_to_plot.center())
                        self.renderer.render(painter, rect_to_plot)

                        painter.restore()

                else:
                    if not check_antialiasing(painter): # only draw this in figure building mode, not in production
                        if self.filename is None: # if the file has a name and no image was produced then there is an error somewhere and the user needs to enter custom code to handle things
                            if self.isText:
                                text = 'Label'
                                pen = QPen(QColor(200, 255, 64))  # yellow
                            else:
                                if ((self.img is None or isinstance(self.img, QRectF)) and self.custom_loading_script):
                                    text = "Error"
                                    pen = QPen(QColor(255, 0, 0))  # red
                                else:
                                    text = "Empty"
                                    pen = QPen(QColor(255, 0, 255))  # pu
                        else:
                            if self.filename == '@template@':
                                text = "Template"
                                pen = QPen(QColor(255, 165, 0))  # orange
                            else:
                                text = "Error"
                                pen = QPen(QColor(255, 0, 0))  # red
                        # TODO deactivate this for production but keep it for image editing!

                        pen.setWidthF(6)
                        painter.setPen(pen)
                        painter.drawRect(rect_to_plot)  # this fills the rect I guess I rather wanna have a pen rather than a brush and a --> probably not what I want !!!

                        font = painter.font()  # Use the current font or set a custom font if desired

                        # Calculate the text's center position within the rectangle
                        font_metrics = QFontMetrics(font)
                        text_width = font_metrics.width(text)
                        text_height = font_metrics.height()
                        text_x = rect_to_plot.x()+ rect_to_plot.width() / 2. - text_width / 2.
                        text_y = rect_to_plot.y()+text_height# + rect_to_plot.height() / 2. + text_height / 2.+text_height # to make it more visible

                        # Draw the text in the middle of the rectangle
                        painter.drawText(int(text_x), int(text_y), text)
            except:
                traceback.print_exc()

            if restore_painter_in_the_end:
                painter.restore()

    def setFigure(self, figure):
        self.must_update_figure_on_first_paint = False

        if figure is not None and isinstance(figure, plt.Figure):
            self.img = figure
            buffer = self._toBuffer(bufferType='svg')
            self.renderer = QSvgRenderer(buffer.read())
            buffer.close()
            if self.width() == 0:
                size = self.renderer.defaultSize()
                self.setWidth(size.width())
                self.setHeight(size.height())
        elif figure is not None and isinstance(figure, str):  # path to an svg file
            self.renderer = QSvgRenderer(figure)  # data --> need convert data to be able to read it
            self.img= self.renderer
            self.filename = figure
            if self.width() == 0:
                size = self.renderer.defaultSize()
                self.setWidth(size.width())
                self.setHeight(size.height())
                if size.width() <= 0:
                    logger.error('image "' + str(self.filename) + '" could not be loaded')
                    self.isSet = False
                    return
        else:
            logger.error(
                'The provided figure is not a valid matplotlib figure nor a valid svg file! Nothing can be done with it... Sorry...')

    def draw_annotations(self,painter, draw=True, parent=None,restore_painter_in_the_end=True):
        try:
            extra_space = 1
            # we get the current transform and it should be applied to all objects but the placed objects to which no rotation should be applied
            current_transform = painter.worldTransform()
            empty_transform = copy_transform_without_rotation(current_transform) # QTransform() #

            if self.annotations is not None and self.annotations:
                objects_grouped_by_position = groupby_position( self.annotations)
                for extra in self.annotations:
                    if isinstance(extra, ScaleBar):
                        extra.unit_to_pixel_conversion_factor = self.px_to_unit_conversion_factor
                        extra.update_bar_at_scale(self.scale)
                    try:
                        try:
                            pos = extra.placement.get_position()
                        except:
                            pos = None
                        if pos:
                            if extra.placement.check_position('top'):
                                extra.setTopLeft(extra.x(), self.topLeft().y())
                            if extra.placement.check_position('left'):
                                extra.setTopLeft(self.topLeft().x(), extra.y())
                            if extra.placement.check_position('right'):
                                extra.setTopLeft(self.x() + self.width(all=True) - extra.width(all=True), extra.y())
                            if extra.placement.check_position('bottom'):
                                extra.setTopLeft(extra.x(), self.topLeft().y() + self.height(all=True) - extra.height(
                                    all=True))
                            if extra.placement.check_position('center_h'):
                                extra.setTopLeft(
                                    self.x() + self.width(all=True) / 2. - extra.width(all=True) / 2., extra.y())
                            if extra.placement.check_position('center_v'):
                                extra.setTopLeft(extra.x(), self.y() + self.height(all=True) / 2. - extra.height(
                                    all=True) / 2.)
                    except:
                        traceback.print_exc()

                # now for each group I need stack them --> give it a try
                for group, objects in objects_grouped_by_position.items():
                    if group == ('free', 'relative'):
                        continue
                    else:
                        pack2(extra_space, objects[0].placement, True,
                              *objects)  # TODO --> find packing mode from position --> should be doable --> and can ignore

                for extra in self.annotations:
                    rect_to_plot = self.getRect(all=True)
                    center = rect_to_plot.center()

                    if ('free', 'relative') in objects_grouped_by_position and  extra in objects_grouped_by_position[('free', 'relative')]:
                        # THESE OBJECTS ARE NOT PLACED SO THEY SHOULD BE ROTATED
                        painter.setWorldTransform(current_transform)

                        if self.theta and not (self.crop_left or self.crop_right or self.crop_top or self.crop_bottom):
                            painter.translate(center.x(), center.y())
                            painter.rotate(self.theta)
                            painter.translate(-center.x(), -center.y())
                            painter.translate(self.getRect(all=True).center()-self.getRect(scale=True).center()) # this is required for rotated shapes

                        if not self.theta and (self.crop_left or self.crop_right or self.crop_top or self.crop_bottom):
                            # this fixes the stuff for rotation only but not for translation --> I'm almost there but not fully yet
                            tmp = get_shape_after_rotation_and_crop(self.getRect(raw=True), self.theta, self.crop_left,
                                                                    self.crop_right, self.crop_top, self.crop_bottom)

                            parent = self.getRect(raw=True)
                            parent = QRectF(0, 0, parent.width(), parent.height())

                            if tmp is not None:
                                tmp2 = parent.topLeft() - QRectF(tmp.boundingRect()).topLeft()
                                painter.translate(tmp2.x() / self.scale, tmp2.y() / self.scale)

                        if self.theta and (self.crop_left or self.crop_right or self.crop_top or self.crop_bottom):
                                # painter.setClipping(False)

                                # TODO --> maybe code this below in getRect(raw_no_crop) for example
                                tmp_crop_left = self.crop_left
                                tmp_crop_right = self.crop_right
                                tmp_crop_top = self.crop_top
                                tmp_crop_bottom = self.crop_bottom

                                self.crop_left =0
                                self.crop_right=0
                                self.crop_top=0
                                self.crop_bottom=0

                                rect_uncropped = self.getRect(all=True)
                                rect_uncropped.translate(rect_to_plot.center()-rect_uncropped.center())
                                # rect_uncropped.translate(rect_to_plot.topLeft()-rect_uncropped.topLeft())
                                center = rect_uncropped.center()

                                self.crop_left = tmp_crop_left
                                self.crop_right=tmp_crop_right
                                self.crop_top=tmp_crop_top
                                self.crop_bottom=tmp_crop_bottom

                                extra_user_translation, extra_translation_at_painter = compute_translation_corrections(self, minimal_return=True)
                                painter.translate(extra_translation_at_painter/self.scale)

                                painter.translate(center.x(), center.y())
                                painter.rotate(self.theta)
                                painter.translate(-center.x(), -center.y())

                                if extra_user_translation:
                                    try:
                                        painter.translate(extra_user_translation[0]/self.scale, extra_user_translation[1]/self.scale)
                                    except:
                                        traceback.print_exc()
                    else:
                        if extra.placement is None or not extra.placement.position_to_string():
                            # THIS OBJECT IS NOT PLACED SO IT SHOULD BE ROTATED
                            painter.setWorldTransform(current_transform)
                            if self.theta:
                                painter.translate(center.x(), center.y())
                                painter.rotate(self.theta)
                                painter.translate(-center.x(), -center.y())
                                painter.translate(self.getRect(all=True).center() - self.getRect(scale=True).center())  # this is required for rotated shapes
                        else:
                            # THIS OBJECT IS PLACED AND THEREFORE SHOULD NOT BE ROTATED
                            painter.setWorldTransform(empty_transform)

                    extra.draw(painter=painter, parent=self)  # added parent for relative plotting
        except:
            traceback.print_exc()

        if restore_painter_in_the_end:
            painter.restore()
    def draw(self, painter, draw=True, parent=None):
        if draw:
            # TODO --> shall I recover and reuse the rect ???

            self.draw_bg(painter, draw=draw, parent=parent, restore_painter_in_the_end=False)

                # letter is good
                # then position all of them as a chain --> TODO
            self.draw_annotations(painter, draw=draw, parent=parent, restore_painter_in_the_end=False)

            painter.restore()

    def setToWidth(self, width_in_px):
        pure_image_width = self.width(scaled=False)# need original height and with in fact
        scale = width_in_px / pure_image_width
        self.scale = scale

    def setToHeight(self, height_in_px):
        pure_image_height = self.height(scaled=False)
        scale = height_in_px/pure_image_height
        self.scale = scale

    def crop(self, left=None, right=None, top=None, bottom=None, all=None):
        if left is not None:
            self.crop_left = left
        if right is not None:
            self.crop_right = right
        if top is not None:
            self.crop_top = top
        if bottom is not None:
            self.crop_bottom = bottom
        if all is not None:
            self.crop_left = all
            self.crop_right = all
            self.crop_top = all
            self.crop_bottom = all
    def set_to_translation(self, translation):
        self.translation = translation

    def boundingRect(self, scaled=True):
 # en fait pas good besoin de prendre les crops et le scale en compte
        # is
        # rect_to_plot = self.adjusted(self.crop_left, self.crop_top, -self.crop_right, -self.crop_bottom)
        rect_to_plot = self.adjusted(0, 0, -self.crop_right-self.crop_left, -self.crop_bottom-self.crop_top)
        # rect_to_plot = self.adjusted(-self.crop_left, -self.crop_top, -self.crop_right, -self.crop_bottom)
        # rect_to_plot = self.adjusted(0,0,0,0)
        # print('begin rect_to_plot', rect_to_plot, self.scale)
        # if kwargs['draw']==True or kwargs['fill']==True:
        # if self.scale is None or self.scale==1:
        #     painter.drawRect(self)
        # else:
        # on clone le rect
        if self.scale is not None and self.scale != 1 and scaled:
            # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
            new_width = rect_to_plot.width() * self.scale
            new_height = rect_to_plot.height() * self.scale
            # print(rect_to_plot.width(), rect_to_plot.height())  # here ok
            # setX changes width --> why is that

            # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
            # rect_to_plot.setX(rect_to_plot.x() * self.scale)
            # rect_to_plot.setY(rect_to_plot.y() * self.scale)
            # maybe to avoid bugs I should use translate instead rather that set x but ok anyways
            # print(rect_to_plot.width(), rect_to_plot.height())# bug here --> too big

            # print(new_height, new_height, self.width(), self.scale, self.scale* self.width())
            rect_to_plot.setWidth(new_width)
            rect_to_plot.setHeight(new_height)
        return rect_to_plot

    SVG_INKSCAPE = 96
    SVG_ILLUSTRATOR = 72

    qualities = [QPainter.SmoothPixmapTransform, QPainter.TextAntialiasing, None, QPainter.Antialiasing] # None sets antialiasing to False -−> see code after # painter.setRenderHint  QPainter.HighQualityAntialiasing,

    '''
    Here are some common interpolation modes:

Qt.SmoothPixmapTransform: This mode enables smooth transformation of pixmaps when they are scaled or rotated. It uses bilinear or bicubic interpolation to provide smoother results, which can be useful when resizing images or drawing rotated content. However, this mode may have a performance impact, especially when dealing with large images or complex transformations.
Qt.HighQualityAntialiasing: This mode enables high-quality antialiasing for painting operations. Antialiasing helps to reduce the jagged appearance of diagonal lines and curves by smoothing their edges. This mode can be useful when drawing shapes, text, or other graphical elements that require smooth edges. However, it may also have a performance impact, especially when dealing with complex scenes or large-scale drawings.
Qt.TextAntialiasing: This mode enables antialiasing specifically for text rendering. It can help to improve the readability and appearance of text in your application, especially at small font sizes or when using custom fonts. However, it may also have a performance impact, especially when dealing with large amounts of text or complex text layouts.

painter = QPainter(some_widget)
painter.setRenderHint(Qt.SmoothPixmapTransform)
# or
painter.setRenderHint(Qt.HighQualityAntialiasing)
# or
painter.setRenderHint(Qt.TextAntialiasing)

painter.setRenderHints(Qt.SmoothPixmapTransform | Qt.HighQualityAntialiasing)
    '''
    def save(self, path, filetype=None, title=None, description=None, svg_dpi=SVG_INKSCAPE, quality=qualities[-1]):
        # if path is None or not isinstance(path, str):
        #     logger.error('please provide a valide path to save the image "' + str(path) + '"')
        #     return
        if path is None:
            filetype = '.tif'

        if filetype is None:
            if path.lower().endswith('.svg'):
                filetype = 'svg'
            else:
                filetype = os.path.splitext(path)[1]
        dpi = 72  # 300 # inkscape 96 ? check for illustrator --> check

        if filetype == 'svg':
            generator = QSvgGenerator()
            generator.setFileName(path)
            if svg_dpi == self.SVG_ILLUSTRATOR:
                generator.setSize(QSize(595, 842))
                generator.setViewBox(QRect(0, 0, 595, 842))
            else:
                generator.setSize(QSize(794, 1123))
                generator.setViewBox(QRect(0, 0, 794, 1123))

            if title is not None and isinstance(title, str):
                generator.setTitle(title)
            if description is not None and isinstance(description, str):
                generator.setDescription(description)
            generator.setResolution(
                svg_dpi)  # fixes issues in inkscape of pt size --> 72 pr illustrator and 96 pr inkscape but need change size

            painter = QPainter(generator)

            # print(generator.title(), generator.heightMM(), generator.height(), generator.widthMM(),
            #       generator.resolution(), generator.description(), generator.logicalDpiX())
        else:
            scaling_factor_dpi = 1
            # scaling_factor_dpi = self.scaling_factor_to_achieve_DPI(300)

            # in fact take actual page size ??? multiplied by factor
            # just take real image size instead


            # image = QtGui.QImage(QSize(self.cm_to_inch(21) * dpi * scaling_factor_dpi, self.cm_to_inch(29.7) * dpi * scaling_factor_dpi), QtGui.QImage.Format_RGBA8888) # minor change to support alpha # QtGui.QImage.Format_RGB32)

            # NB THE FOLLOWING LINES CREATE A WEIRD ERROR WITH WEIRD PIXELS DRAWN some sort of lines NO CLUE WHY

            img_bounds = self.boundingRect()
            # image = QtGui.QImage(QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height()* scaling_factor_dpi),  QtGui.QImage.Format_RGBA8888)  # minor change to support alpha # QtGui.QImage.Format_RGB32)
            image = QtGui.QImage(QSize(int(img_bounds.width() * scaling_factor_dpi), int(img_bounds.height()* scaling_factor_dpi)),  QtGui.QImage.Format_RGBA8888)  # minor change to support alpha # QtGui.QImage.Format_RGB32)
            # image = QtGui.QImage(QSize(int(img_bounds.scale(scaling_factor_dpi))),  QtGui.QImage.Format_RGBA8888)  # minor change to support alpha # QtGui.QImage.Format_RGB32)
            # print('size at dpi',QSize(img_bounds.width() * scaling_factor_dpi, img_bounds.height()* scaling_factor_dpi))
            # QSize(self.cm_to_inch(0.02646 * img_bounds.width())
            # self.cm_to_inch(0.02646 * img_bounds.height())
            # need convert pixels to inches
            # is there a rounding error

            # force white bg for non jpg
            try:
                # print(filetype.lower())
                # the tif and png file formats support alpha
                if not filetype.lower() == '.png' and not filetype.lower() == '.tif' and not filetype.lower() == '.tiff':
                    image.fill(QColor.fromRgbF(1,1,1))
                else:
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # image.fill(QColor.fromRgbF(1, 1, 1, alpha=1))
                    # TODO KEEP in fact image need BE FILLED WITH TRANSPARENT OTHERWISE GETS WEIRD DRAWING ERRORS
                    # TODO KEEP SEE https://stackoverflow.com/questions/13464627/qt-empty-transparent-qimage-has-noise
                    # image.fill(qRgba(0, 0, 0, 0))
                    image.fill(QColor.fromRgbF(0,0,0,0))
            except:
                pass
            painter = QPainter(image)  # see what happens in case of rounding of pixels
            # painter.begin()
            painter.scale(scaling_factor_dpi, scaling_factor_dpi)
        if quality is not None:
            painter.setRenderHint(quality)  # to improve rendering quality
        else:
            painter.setRenderHint(QPainter.Antialiasing, False)  # to improve rendering quality
        self.draw(painter)
        painter.end()
        if path is None:
            return image
        if filetype != 'svg':
            image.save(path)
            return image

    #based on https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
    def convert_qimage_to_numpy(self, qimage):
        qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = qimage.width()
        height = qimage.height()
        image_pointer = qimage.bits() # creates a deep copy --> this is what I want
        # image_pointer.setsize(qimage.byteCount())
        try:
            image_pointer.setsize(qimage.sizeInBytes()) # qt6 version of the stuff
        except:
            image_pointer.setsize(qimage.byteCount())
        # arr = np.array(image_pointer,copy=True).reshape(height, width, 4)
        arr = np.array(image_pointer).reshape(height, width, 4)
        arr = arr[..., 0:3]
        arr = RGB_to_BGR(arr)  # that seems to do the job
        return arr

    def draw_inner_layout_selection(self, painter):
        painter.save()
        try:
            rect = self.getRect(all=True) # there is a big bug with that
            painter.setPen(QPen(QColor(0, 255, 255)))
            painter.drawRect(rect)
        except:
            print('inner error')
            traceback.print_exc()
        painter.restore()

    def width(self, *args, **kwargs):
        # this should be uncropped and unrotated in fact or rotated in fact and uncropped ??? -−> think about it
        if args or kwargs:
            return self.getRect(all=True).width()  # this is probably incorrect but ok for now
        else:
            return self.getRect().width() #* (self.scale if self.scale else 1)

    def height(self, *args, **kwargs):
        if args or kwargs:
            return self.getRect(all=True).height()  # this is probably incorrect but ok for now
        else:
            return self.getRect().height()# * (self.scale if self.scale else 1)

    def getRect(self, *args, **kwargs):
        # w = super().width()
        # h = super().height()
        x = self.x()
        y = self.y()
        # if isinstance(self.img, np.ndarray):
        #     w = self.img.shape[1]
        #     h = self.img.shape[0]
        # elif isinstance(self.img, plt.Figure):
        #     rect = get_fig_rect(self.img)
        #     w = rect.width()
        #     h = rect.height()
        # elif isinstance(self.renderer, QSvgRenderer):
        #     size = self.renderer.defaultSize()
        #     w = size.width()
        #     h = size.height()
        w, h = self.get_raw_size()

        # if self.is_template():
        #     return QRectF(x,y,w,h)

        if args or kwargs:

            if 'raw' in kwargs:
                return QRectF(x,y,w,h)
            if 'all' in kwargs:
                rect_to_plot = QRectF(x, y, w, h)


                center = rect_to_plot.center()

                if True:  # try to correct the stuff in fact not really working --> see how I can fix that
                    tmp = QRectF(rect_to_plot)
                    if self.crop_left:
                        tmp.setWidth(tmp.width() - self.crop_left)
                    if self.crop_right:
                        tmp.setWidth(tmp.width() - self.crop_right)
                    if self.crop_top:
                        tmp.setHeight(tmp.height() - self.crop_top)
                    if self.crop_bottom:
                        tmp.setHeight(tmp.height() - self.crop_bottom)
                    tmp.setWidth(tmp.width() / self.scale)
                    tmp.setHeight(tmp.height() / self.scale)

                if self.theta and not self.theta == 0:  # with that the rotation is ok but the position of the rect is not --> but with angle 0 it is --> can I correct for the translation
                    t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
                                                                                                    -center.y())  # .translate(center.x(), center.y())
                    rotatedPolygon = t.mapToPolygon(rect_to_plot.toRect())  # Map the rectangle to a rotated polygon
                    # Create a QRectF from the bounding rectangle of the rotated polygon
                    rotatedRect = QRectF(rotatedPolygon.boundingRect())
                else:
                    if self.crop_left:
                        rect_to_plot.setWidth(rect_to_plot.width() - self.crop_left)
                    if self.crop_right:
                        rect_to_plot.setWidth(rect_to_plot.width() - self.crop_right)
                    if self.crop_top:
                        rect_to_plot.setHeight(rect_to_plot.height() - self.crop_top)
                    if self.crop_bottom:
                        rect_to_plot.setHeight(rect_to_plot.height() - self.crop_bottom)
                    rect_to_plot.setWidth(rect_to_plot.width() / self.scale)
                    rect_to_plot.setHeight(rect_to_plot.height() / self.scale)

                    return rect_to_plot
                try:
                    rotatedRect = QRectF(rotatedRect.boundingRect())

                except:
                    pass

                if self.crop_left:
                    rotatedRect.setWidth(rotatedRect.width() - self.crop_left)
                if self.crop_right:
                    rotatedRect.setWidth(rotatedRect.width() - self.crop_right)
                if self.crop_top:
                    rotatedRect.setHeight(rotatedRect.height() - self.crop_top)
                if self.crop_bottom:
                    rotatedRect.setHeight(rotatedRect.height() - self.crop_bottom)

                rotatedRect.setWidth(rotatedRect.width() / self.scale)
                rotatedRect.setHeight(rotatedRect.height() / self.scale)
                rotatedRect.translate(tmp.topLeft() - rotatedRect.topLeft())

                return rotatedRect
            else:
                rect_to_plot = QRectF(self.x(), self.y(), w, h)
                if self.crop_left:
                    rect_to_plot.setWidth(rect_to_plot.width() - self.crop_left)
                if self.crop_right:
                    rect_to_plot.setWidth(rect_to_plot.width() - self.crop_right)
                if self.crop_top:
                    rect_to_plot.setHeight(rect_to_plot.height() - self.crop_top)
                if self.crop_bottom:
                    rect_to_plot.setHeight(rect_to_plot.height() - self.crop_bottom)
                rect_to_plot.setWidth(rect_to_plot.width() / self.scale)
                rect_to_plot.setHeight(rect_to_plot.height() / self.scale)
                return rect_to_plot
        else:
            rect_to_plot = QRectF(self.x(), self.y(), w, h)
            center = rect_to_plot.center()
            rotatedRect = rect_to_plot
            if self.theta:
                t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
                                                                                                -center.y())
                rotatedPolygon = t.mapToPolygon(rect_to_plot.toRect())  # Map the rectangle to a rotated polygon
                # Create a QRectF from the bounding rectangle of the rotated polygon
                rotatedRect = QRectF(rotatedPolygon.boundingRect())
            if self.crop_left:
                rotatedRect.setWidth(rotatedRect.width() - self.crop_left)
            if self.crop_right:
                rotatedRect.setWidth(rotatedRect.width() - self.crop_right)
            if self.crop_top:
                rotatedRect.setHeight(rotatedRect.height() - self.crop_top)
            if self.crop_bottom:
                rotatedRect.setHeight(rotatedRect.height() - self.crop_bottom)
            try:
                return QRectF(rotatedRect.boundingRect())
            except:
                return rotatedRect

    # def contains(self, *__args):
    #     try:
    #         # if self.__sel:
    #         #     return self.__sel.contains(*__args)
    #         if __args is not None and __args:
    #             return self.getRect(all=True).contains(*__args)
    #         else:
    #             return False
    #         # return self.getRect(scale=True).contains(*__args)
    #     except:
    #         # traceback.print_exc() # somehow this generates an error but does it make sense
    #         # I have too many handling of rotations --> that may cause trouble too
    #         # simplest is to really make the rect as it should be and as selection should be !!!
    #         pass
    #     return False


    def __contains__(self, item):
        return False

    def __copy__(self):
        return self # TODO --> rather do a clone of self

    def __deepcopy__(self, memo):
        return self # TODO --> rather do a clone of self

    # DEV NOTE KEEP be extremely careful with that because this is what is compared when using == !!! so if poorly implemented this can have dramatic effects
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        class_name = type(self).__name__
        memory_address = hex(id(self))

        # print('debugggging', self.img, self.custom_loading_script, ((self.img is None or isinstance(self.img, QRectF)) and self.filename is not None), ((self.img is None or isinstance(self.img, QRectF)) and self.custom_loading_script))

        if self.isText:
            class_name = 'Label_' + class_name
        elif self.is_template():
            class_name = 'Template_' + class_name
        # elif (self.img is None and self.filename is not None) or (self.img is None and self.custom_loading_script):
        elif ((self.img is None or isinstance(self.img, QRectF)) and self.filename is not None) or ((self.img is None or isinstance(self.img, QRectF)) and self.custom_loading_script):
            class_name = 'Error_' + class_name
        elif isinstance(self.img, plt.Figure):
            class_name = 'Fig_' + class_name
        elif self.renderer is not None:
            class_name = 'SVG_'+class_name
        elif self.img is None:
            class_name = 'Empty_'+class_name

        if self.filename is not None:
            base_name = os.path.basename(self.filename)
            return f"{class_name}-{memory_address}-{base_name}"
        return f"{class_name}-{memory_address}"

    def to_dict(self):
        x = self.x()
        y = self.y()
        width = self.width()
        height = self.height()

        # Create a dictionary representation of the values of the super object
        output_dict = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            # 'args':self.filename,
            'filename':self.filename,
        }
        # Update the dictionary with the __dict__ of Rectangle2D
        output_dict.update(self.__dict__)
        try:
            output_dict['extra_user_translation']=list(self.extra_user_translation) #
        except:
            pass
        try:
            del output_dict['script_log']
        except:
            pass
        if isinstance(self.img, QRectF):
            output_dict['template_rect']=self.img
        output_dict['LUTs']=str(self.LUTs)
        output_dict['channels']=str(self.channels)
        output_dict['lower_indices']=None
        output_dict['upper_indices']=None
        output_dict['lower_values']=None
        output_dict['upper_values']=None
        # output_dict['theta']=self.theta
        return output_dict

    def execute_code_unsafe(self, code, parent=None): # TODO --> see how to handle the security!!!
        # Save the original streams
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        # Create temporary streams
        temp_stdout = io.StringIO()
        temp_stderr = io.StringIO()

        # Redirect the streams to the temporary buffers
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

        # If an exception occurs, capture it and include it in the output
        local_namespace = {'plt': plt, 'Img': Img, 'np':np} # all imports should be made there (otherwise the user needs to do the imports himslef --> painful though
        try:
            if self.filename and (self.filename.strip().startswith("['") and self.filename.strip().endswith("']")):
                self.filename = reload_string_list(self.filename)
            exec(code, locals(), local_namespace)
            if 'self.img' in local_namespace:
                self.img = local_namespace['self.img'] # shall I transfer more things ? -−> maybe !!!
        except Exception as e:
            print("Error:", e)
            temp_stderr.write(f"An error occurred: {str(e)}\n")
            temp_stderr.write(f"{traceback.format_exc()}\n")

        # Restore the original streams
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

        # Return the captured output
        # return temp_stdout.getvalue(), temp_stderr.getvalue()
        # we redirect the log and maybe the output to a file
        self.script_log = temp_stderr.getvalue() if temp_stderr.getvalue() else temp_stdout.getvalue()
        # print('#'*20)
        # print('self.script_log', self.script_log)
        # print('-' * 20)
        # print('dabu', temp_stdout.getvalue())
        # print('%' * 20)
        # print('dabi',temp_stderr.getvalue())
        # print('#' * 20)

        # Close the temporary streams
        temp_stdout.close()
        temp_stderr.close()

        if self.filename:
            if not isinstance(self.filename, str) and isinstance(self.filename, list):
                self.filename= str(self.filename)


if __name__ == '__main__':
    app = QApplication(sys.argv)# IMPORTANT KEEP !!!!!!!!!!!
    import sys

    if True:
        img1 = Image2D('/E/Sample_images/counter/01.png', user_comments='comments added by the user test')
        print('before rotation',img1.getRect(), img1.getRect(scale=True), img1.getRect(all=True))

        img1.crop(10,15,20,30)
        print('after crop', img1.getRect(), img1.getRect(scale=True), img1.getRect(all=True)) # -> all is ok --> so maybe just the drawing is fucked in fact now

        img1.crop(0, 0, 0, 0)
        img1.set_rotation(45)
        print('after rotation',img1.getRect(), img1.getRect(scale=True), img1.getRect(all=True)) # there is a bug in scale because it does not apply rotation --> probably not correct

        img1.crop(10,15,20,30)
        print('after rotation and crop', img1.getRect(), img1.getRect(scale=True), img1.getRect(all=True))  # there is a bug i
        sys.exit(0)

    if True:
        # Data for plotting
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        fig, ax = plt.subplots()
        print(type(fig))
        ax.plot(t, s)

        ax.set(xlabel='time (s)', ylabel='voltage (mV)', title=None)
        ax.grid()

        figure = Image2D(fig)



        sys.exit(0)

    if True:
        # todo test the use of an empty image with given dimensions then once this is done try to do stuff
        # in fact in all cases the only thing that matters is the initial rect --> I could put a rect instead of an image and that would give me all the things I need
        # TODO

        # try to do that
        try:
            empty_img = Image2D()
            print(empty_img.width())
        except:
            traceback.print_exc()
            print('test that empty images are not allowed')

        # do I already have a construction that would do the job
        empty_img = Image2D(width=120, height=240)# that seems to work --> but super is never called --> see
        # maybe allow an image to be initialized by a rect (to act as an empty image and if that is the case the image could be replaced) by what I need !!!

        print(empty_img)
        print(empty_img.width())

        sys.exit(0)

    # is it possible that I never implemented the rotation ???
    # can I do it???

    # VERY GOOD --> I HAVE FINALLY FIXED IT

    # ça marche --> voici deux examples de shapes
    test = Image2D(x=12, y=0, width=100, height=100)  # could also be used to create empty image with

    print(test.img)
    print(test.boundingRect())
    # print(test.get_P1().x())

    # bug qd on definit une image comme param
    # test = Image2D('/E/Sample_images/counter/06.png')
    test = Image2D('/E/Sample_images/counter/01.png')
    print(test.boundingRect())  # --> it is ok there so why not below # not callable --> why -->
    # print(test.get_P1())  # ça marche donc où est le bug
    # print(test.get_P1().y())  # ça marche donc où est le bug
    # print(test.getP1().width())
    # ça marche

    # try draw on the image the quivers
    # img0.setLettering('<font color="red">A</font>')
    # # letter
    # img0.annotation.append(Rect2D(88, 88, 200, 200, stroke=3, color=0xFF00FF))
    # img0.annotation.append(Ellipse2D(88, 88, 200, 200, stroke=3, color=0x00FF00))
    # img0.annotation.append(Circle2D(33, 33, 200, stroke=3, color=0x0000FF))

    test.annotations.append(TAText2D('<font color="red">A</font>', placement='top-left'))

    test.annotations.append(Line2D(33, 33, 88, 88, stroke=3, color=0x0000FF))

    test.annotations.append(Line2D(128, 33, 88, 88, stroke=0.65, color=0xFFFF00))
    # img0.annotation.append(Freehand2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
    # # img0.annotation.append(PolyLine2D(10, 10, 20, 10, 20, 30, 288, 30, color=0xFFFF00, stroke=3))
    # img0.annotation.append(Point2D(128, 128, color=0xFFFF00, stroke=6))

    # add a save to the image --> so that it exports as a raster --> TODO


    # painter =
    # img = test.draw() # if no painter --> create one as not to lose any data and or allow to save as vectorial


    img = test.save('/E/trash/test_line2D.tif')

    # test.save('/E/Sample_images/sample_images_PA/mini_vide/analyzed/trash/test_line2D.svg')

    #trop facile --> just hack it so that it can return a single qimage # or return a numpy image that is then plotted -> should not be too hard !!! I think --> TODO
    img = test.convert_qimage_to_numpy(img) # --> ok but I just need to swap the channels then I'll be done --> try that maybe with plenty of input images just to see

    # empty image work --> now try to see if I can create an image from a Figure and or a vector graphic ??? -−> TODO

    # shall I save
    # try with non RGB images just to see

    # img = RGB_to_BGR(img)

    # almost there --> just need to check that the size of the image is ok and that everything is fine
    plt.imshow(img)
    plt.show()


    print('here')
    img2 = Image2D('/E/Sample_images/counter/02.png')
    print('here3')
    preview(img2)

    # test replace the plot of pyTA of the polarity by this one --> should be quite easy TODO I think --> TODO

    # ok --> it all seems to work --> see how I can handle that



    # --> all seems ok now
    # --> put this in the advanced sql plotter

