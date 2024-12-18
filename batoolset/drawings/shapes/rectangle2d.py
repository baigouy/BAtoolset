# this is the fully fixed version with the new rect
# THIS IS THE NEW VERSION OF THE CODE BUT IN ORDER NOT TO BREAK STUFF I WILL KEEP IT LIKE THAT

from batoolset.settings.global_settings import set_UI  # set the UI to qtpy
set_UI()
import os
import sys
import traceback
from qtpy.QtCore import QPointF, QRectF, Qt, QRect
from qtpy.QtGui import QBrush, QPen, QColor, QTransform
from batoolset.tools.logger import TA_logger

logger = TA_logger()

# TODO --> try to simplify that and finalize it once for good
# rather use a dict as **kwargs to set opacity and alike --> that would allow me to handle much more flexibly my tools -−> this is not possible in fact possible but not very beautiful
'''
def __init__(self, *args, **kwargs):
    # set default values for keyword arguments
    color = kwargs.get('color', 0xFFFF00)
    stroke = kwargs.get('stroke', 0.65)
'''

class Rectangle2D(QRectF):
    """
    A custom class for 2D rectangles with additional features such as color, fill color, opacity, stroke, line style,
    rotation, scale, and translation.

    Args:
        *args: Variable-length arguments that can represent the rectangle coordinates.
        color (int): Color of the rectangle in hexadecimal format (default: 0xFFFF00).
        fill_color (int): Fill color of the rectangle in hexadecimal format (default: None).
        opacity (float): Opacity value between 0.0 and 1.0 (default: 1.0).
        stroke (float): Stroke width of the rectangle (default: 0.65).
        line_style (Union[None, int, List[int]]): Line style of the rectangle. None for a solid line,
            or any of the following values: Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine.
            If a list of integers is provided, it represents a custom dash pattern (default: None).
        theta (float): Rotation angle in degrees (default: 0.0).
        **kwargs: Additional keyword arguments.

    Attributes:
        isSet (bool): Indicates whether the rectangle is set.
        color (int): Color of the rectangle in hexadecimal format.
        fill_color (int): Fill color of the rectangle in hexadecimal format.
        stroke (float): Stroke width of the rectangle.
        opacity (float): Opacity value between 0.0 and 1.0.
        scale (float): Scaling factor of the rectangle (default: 1.0).
        translation (QPointF): Translation of the rectangle.
        line_style (Union[None, int, List[int]]): Line style of the rectangle.

    """

    def __init__(self, *args,x=0, y=0, width=0, height=0, color=0xFFFFFF, fill_color=None, opacity=1., stroke=0.65, line_style=None, theta=0, __version__=1.0,**kwargs):
        if args:
            super(Rectangle2D, self).__init__(*args)
        else:
            super(Rectangle2D, self).__init__(x,y,width,height)

        # if not args:
        #     self.isSet = False
        # else:
        #     self.isSet = True
        self.color = color
        self.fill_color = fill_color
        self.stroke = stroke
        self.opacity = opacity
        self.isSet = False
        self.scale = 1
        # self.translation = QPointF()
        self.line_style = line_style
        self.theta = theta
        self.incompressible_width=0
        self.incompressible_height=0
        self.immutable = False
        self.__init_called__ = True # this is just to be used to prevent non initialized images to exist
        # self.__sel = None
        self.__version__ = __version__

    def set_rotation(self, theta):
        """
        Set the rotation angle of the rectangle.

        Args:
            theta (float): Rotation angle in degrees.

        """
        self.theta = theta

    def set_opacity(self, opacity):
        """
        Set the opacity value of the rectangle.

        Args:
            opacity (float): Opacity value between 0.0 and 1.0.

        """
        self.opacity = opacity

    def set_line_style(self,style):
        """
        allows lines to be dashed or dotted or have custom pattern

        :param style: a list of numbers or any of the following Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine but not Qt.CustomDashLine, Qt.CustomDashLine is assumed by default if a list is passed in. None is also a valid value that resets the line --> assume plain line
        :return:
        """
        self.line_style = style
        # if style is a list then assume custom pattern otherwise apply solidline


    def draw(self, painter, parent=None, **kwargs):
        # print(kwargs)
        # if kwargs is not None and kwargs:
        #     pass
        # else:
        #     return
        # self.__sel = None
        if self.color is None and self.fill_color is None:
            return

        painter.save()

        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.NoBrush)

        painter.setOpacity(self.opacity)





        if self.color is not None:
            pen = QPen(QColor(self.color))
            if self.stroke is not None:
                pen.setWidthF(self.stroke)
            if self.line_style is not None:
                if self.line_style in [Qt.SolidLine, Qt.DashLine, Qt.DashDotLine, Qt.DotLine, Qt.DashDotDotLine]:
                    pen.setStyle(self.line_style)
                elif isinstance(self.line_style, list):
                    pen.setStyle(Qt.CustomDashLine)
                    pen.setDashPattern(self.line_style)
            pen.setJoinStyle(Qt.MiterJoin)# RoundJoin #BevelJoin # some day offer the handling of that but later
            # if kwargs['draw']==True:

            # pen.setStyle(Qt.DashLine)
            # pen.setStyle(Qt.CustomDashLine)
            # pen.setDashPattern([1, 4, 5, 4])

            painter.setPen(pen)
        # if kwargs['fill']==True:
        else:
            painter.setPen(Qt.NoPen) # required to draw something filled without a border

        # needed to have a filling color in a shape without contour
        if self.fill_color is not None:
            painter.setBrush(QBrush(QColor(self.fill_color)))

        # rect_to_plot = self.adjusted(0,0,0,0)
        rect_to_plot = self.getRect(scale=True).adjusted(0,0,0,0)

        if parent is not None and parent.scale is not None and parent.scale != 1:
            rect_to_plot = self.__get_scaled_rect(rect_to_plot, 1. / parent.scale)

        if parent is not None:
            rect_to_plot = rect_to_plot.translated(parent.topLeft())

        # print('rect_to_plot',rect_to_plot)

        # print('begin rect_to_plot', rect_to_plot, self.scale)
        # if kwargs['draw']==True or kwargs['fill']==True:
        # if self.scale is None or self.scale==1:
        #     painter.drawRect(self)
        # else:
            # on clone le rect
        # old crappy code --> forget and clean
        # if self.scale is not None and self.scale != 1:
        #     # TODO KEEP THE ORDER THIS MUST BE DONE THIS WAY OR IT WILL GENERATE PLENTY OF BUGS...
        #     new_width = rect_to_plot.width() * self.scale
        #     new_height = rect_to_plot.height() * self.scale
        #     # print(rect_to_plot.width(), rect_to_plot.height())  # here ok
        #     # setX changes width --> why is that
        #
        #     # TODO BE EXTREMELY CAREFUL AS SETX AND SETY CAN CHANGE WIDTH AND HEIGHT --> ALWAYS TAKE SIZE BEFORE OTHERWISE THERE WILL BE A PB AND ALWAYS RESET THE SIZE WHEN SETX IS CALLED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     # Sets the left edge of the rectangle to the given x coordinate. May change the width, but will never change the right edge of the rectangle. --> NO CLUE WHY SHOULD CHANGE WIDTH THOUGH BUT BE CAREFUL!!!
        #     rect_to_plot.setX(rect_to_plot.x()*self.scale)
        #     rect_to_plot.setY(rect_to_plot.y()*self.scale)
        #     # maybe to avoid bugs I should use translate instead rather that set x but ok anyways
        #     # print(rect_to_plot.width(), rect_to_plot.height())# bug here --> too big
        #
        #     # print(new_height, new_height, self.width(), self.scale, self.scale* self.width())
        #     rect_to_plot.setWidth(new_width)
        #     rect_to_plot.setHeight(new_height)

        # print('mid rect_to_plot', rect_to_plot)
        # if self.translation is not None:
        #
        #     # rect_to_plot.setX(rect_to_plot.x()+self.translation.x())
        #     # rect_to_plot.setY(rect_to_plot.y()+self.translation.y())
        #     rect_to_plot.translate(self.translation)

        # print('rect to plot', rect_to_plot)

        # if self.color is not None:

        # rotation marche mais position est pas du tout bonne --> voir comment gérer ça ??
        # center = rect_to_plot.center()
        # painter.translate(center.x(), center.y())
        # painter.rotate(45)
        # rct = QRectF(-center.x(), -center.y(), rect_to_plot.width(), rect_to_plot.height())
        # painter.drawRect(rct)
        # rotatedRect = rect_to_plot

        if self.theta is not None and self.theta != 0:
            # this is only if a rotation should be applied
            painter.translate(rect_to_plot.center())
            # print('center', rect_to_plot.center())
            # painter.setWorldTransform()
            # print(painter.combinedTransform().)



            # TODO use this maybe to get bounds of rotated shape properly...
            # center = rect_to_plot.center()
            # t = QTransform().translate( center.x(), center.y() ).rotate( self.theta ).translate( -center.x(), -center.y() )
            # rotatedRect =  t.mapToPolygon( rect_to_plot.toRect() )  #// mapRect() returns the bounding rect of the rotated rect

            # on dirait que c'est ça faut faire deux translations en x

            painter.rotate(self.theta)

            # ça marche faut deux translations

            # painter.translate(rect_to_plot.width() / 2, -rect_to_plot.height() )


            #see https://www.qtcentre.org/threads/15540-Rotating-a-rectangle-about-its-origin --> maybe ok

            # difference = old center -
            # old_center = self.boundingRect().center()
            # trans = painter.combinedTransform()
            # transformed = trans.mapRect(rect_to_plot)

            # peut etre une piste mais a pas l'air de marcher
            # print(self.boundingRect(), transformed)
            # trans = transformed.center()-self.boundingRect().center()


            # width_dif = transformed.width()-self.boundingRect().width()
            # width_dif/=2
            # print('trans', trans, transformed.center(), self.boundingRect().center(), width_dif)

            # painter.translate(-trans)
            # only if rotation
            painter.translate(-rect_to_plot.center())
        painter.drawRect(rect_to_plot)
        # else:
        #     painter.fillRect(rect_to_plot, QColor(self.fill_color))

        painter.restore()

        # self.__sel = rotatedRect
        # if rotatedRect:
        #     if isinstance(rotatedRect, QRectF):
        #         return rotatedRect
        #     self.__sel =rotatedRect.boundingRect()
        #     return self.__sel
        # else:
        #     return self.__sel


    def __get_scaled_rect(self, rect_to_plot, scale):
        rect = QRectF(rect_to_plot.x()*scale, rect_to_plot.y()*scale, rect_to_plot.width()*scale, rect_to_plot.height()*scale)
        return rect

    # TODO modify that to handle rotation etc...
    def boundingRect(self, scaled=True):
        # should I return the scaled version or the orig --> think about it...
        if scaled:
            # in fact scale only handles rotation and not scale --> the size is therefore wrong -->
            try:
                if self.theta is not None and self.theta!=0:
                    # print('entering')
                    center = self.center()
                    t = QTransform().translate( center.x(), center.y()).rotate(self.theta).translate(-center.x(), -center.y())
                    # print('entersd')
                    rotatedRect = t.mapRect( self )  #// mapRect() returns the bounding rect of the rotated rect
                    # print('rotated',rotatedRect )
                    return rotatedRect
            except:
                pass
        return self

    # def add(self, *args):
    #     point = args[1]
    #     self.setWidth(point.x()-self.x()) # WHY IS THAT A - AND NOT A + ??? DID I DO A MISTAKE ??? −−> COULD BE  # BUT KEEP LIKE THAT JUST IN CASE
    #     self.setHeight(point.y()-self.y())
    #     # self.isSet = True

    # def set_P1(self, *args):
    #     if not args:
    #         logger.error("no coordinate set...")
    #         return
    #     if len(args) == 1:
    #         self.moveTo(args[0].x(), args[0].y())
    #     else:
    #         self.moveTo(QPointF(args[0], args[1]))

    # def get_P1(self):
    #     return QPointF(self.x(), self.y())

    def set_to_scale(self, factor):
        self.scale = factor

    # def set_to_translation(self, translation):
    #     self.translation = translation

    # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    def __eq__(self, other):
        try:
            return hash(str(self)) == other.__hash__()
        except:
            return False

    # hack to make column hashable to be able to add it to a dict, see https://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    def __hash__(self):
        return hash(str(self))

    def getIncompressibleWidth(self):
        return self.incompressible_width

    def getIncompressibleHeight(self):
        return self.incompressible_height

    def get_incompressible_size(self, axis=None):
        # DO I REALLY NEED THAT ANYMORE --> SHALL IT NOT SIMPLY ALWAYS RETURN 0 ???
        if axis:
            if axis == 'X':
                return self.getIncompressibleWidth()
            else:
                return self.getIncompressibleHeight()
        return 0

    # can also be done like that
    # def setTopLeft(self, x_or_rect, y=None):
    #     if y is None:
    def setTopLeft(self, *args):
        if args:
            if len(args)==1:
                # assume a QpointF
                super().moveTopLeft(args[0])
            elif len(args)==2:
                super().moveTopLeft(QPointF(args[0], args[1]))
            else:
                logger.error('invalid args for top left')


    # def getRect(self,*args, **kwargs):
    #     if args or kwargs:
    #         if 'all' in kwargs:
    #             # print('in there!')
    #             # rect_to_plot = QRectF(*super().getRect())#QRectF(self.x(), self.y(), self.width() / self.scale, self.height() / self.scale)
    #             rect_to_plot = QRectF(self.x(), self.y(), self.width() , self.height() )
    #             center = rect_to_plot.center()
    #             if self.theta:
    #                 # t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),-center.y())
    #                 # rotatedRect = t.mapToPolygon(rect_to_plot.toRect())  # // mapRect() returns the bounding rect
    #                 t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
    #                                                                                                 -center.y())
    #                 rotatedPolygon = t.mapToPolygon(rect_to_plot.toRect())  # Map the rectangle to a rotated polygon
    #
    #                 # Create a QRectF from the bounding rectangle of the rotated polygon
    #                 rotatedRect = QRectF(rotatedPolygon.boundingRect())
    #
    #                 if self.crop_left:
    #                     rotatedRect.setWidth(rotatedRect.width()-self.crop_left)
    #                 if self.crop_right:
    #                     rotatedRect.setWidth(rotatedRect.width()-self.crop_right)
    #                 if self.crop_top:
    #                     rotatedRect.setHeight(rotatedRect.height()-self.crop_top)
    #                 if self.crop_bottom:
    #                     rotatedRect.setHeight(rotatedRect.height()-self.crop_bottom)
    #
    #                 rotatedRect.setWidth(rotatedRect.width()/self.scale)
    #                 rotatedRect.setHeight(rotatedRect.height()/self.scale)
    #             else:
    #                 if self.crop_left:
    #                     rect_to_plot.setWidth(rect_to_plot.width() - self.crop_left)
    #                 if self.crop_right:
    #                     rect_to_plot.setWidth(rect_to_plot.width() - self.crop_right)
    #                 if self.crop_top:
    #                     rect_to_plot.setHeight(rect_to_plot.height() - self.crop_top)
    #                 if self.crop_bottom:
    #                     rect_to_plot.setHeight(rect_to_plot.height() - self.crop_bottom)
    #                 rect_to_plot.setWidth(rect_to_plot.width() / self.scale)
    #                 rect_to_plot.setHeight(rect_to_plot.height() / self.scale)
    #                 return rect_to_plot
    #             try:
    #                 return QRectF(rotatedRect.boundingRect())
    #             except:
    #                 return rotatedRect
    #             # return QRectF(self.x(), self.y(), self.width() / self.scale, self.height() / self.scale)
    #         else:
    #             # default to scaled object
    #             return QRectF(self.x(), self.y(), self.width()/self.scale, self.height()/self.scale)
    #     else:
    #         # return super().getRect()
    #         return QRectF(*super().getRect())


    def getRect(self,*args, **kwargs):
        if args or kwargs:
            if 'all' in kwargs:
                # print('in there!')
                # rect_to_plot = QRectF(*super().getRect())#QRectF(self.x(), self.y(), self.width() / self.scale, self.height() / self.scale)
                rect_to_plot = QRectF(self.x(), self.y(), self.width() / self.scale, self.height() / self.scale)
                center = rect_to_plot.center()
                if self.theta:
                    # t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),-center.y())
                    # rotatedRect = t.mapToPolygon(rect_to_plot.toRect())  # // mapRect() returns the bounding rect
                    t = QTransform().translate(center.x(), center.y()).rotate(self.theta).translate(-center.x(),
                                                                                                    -center.y())
                    rotatedPolygon = t.mapToPolygon(rect_to_plot.toRect())  # Map the rectangle to a rotated polygon

                    # Create a QRectF from the bounding rectangle of the rotated polygon
                    rotatedRect = QRectF(rotatedPolygon.boundingRect())
                else:
                    return rect_to_plot
                try:
                    return QRectF(rotatedRect.boundingRect())
                except:
                    return rotatedRect
                # return QRectF(self.x(), self.y(), self.width() / self.scale, self.height() / self.scale)
            else:
                # default to scaled object
                return QRectF(self.x(), self.y(), self.width()/self.scale, self.height()/self.scale)
        else:
            # return super().getRect()
            return QRectF(*super().getRect())

    def width(self,*args, **kwargs):
        if args or kwargs:
            return self.width()/self.scale # this is probably incorrect but ok for now
        else:
            return super().width()
        # return self.getRect(*args, **kwargs).width()

    def height(self,*args, **kwargs):
        if args or kwargs:
            return self.height()/self.scale
        else:
            return super().height()
        # return self.getRect(*args, **kwargs).height()

    def getAR(self, invert=False):
        # try:
            if invert:
                return float(self.height()) / float(self.width())
            return float(self.width()) / float(self.height())
        # except:
        #     return 1.

    def contains(self, *__args):
        try:
            # if self.__sel:
            #     return self.__sel.contains(*__args)
            if __args is not None and __args:
                return self.getRect(all=True).contains(*__args)
            else:
                return False
            # return self.getRect(scale=True).contains(*__args)
        except:
            # traceback.print_exc() # somehow this generates an error but does it make sense
            # I have too many handling of rotations --> that may cause trouble too
            # simplest is to really make the rect as it should be and as selection should be !!!
            pass
        return False

    # def __str__(self):
    #     return f'rectangle {super().__str__()}'
    def set_to_center(self, centroid):
        w = self.width()
        h = self.height()
        if isinstance(centroid, (tuple, list)):

            self.setX(centroid[0]-self.width()/2.)
            self.setY(centroid[1]-self.height()/2.)

        else:
            self.setX(centroid.x()-self.width()/2.)
            self.setY(centroid.y()-self.height()/2.)
        self.setWidth(w)
        self.setHeight(h)
    def add(self, *args):
        point = args[1]
        self.setWidth(point.x()-self.x()) # WHY IS THAT A - AND NOT A + ??? DID I DO A MISTAKE ??? −−> COULD BE  # BUT KEEP LIKE THAT JUST IN CASE
        self.setHeight(point.y()-self.y())
        self.isSet = True

    # def __repr__(self):
    #     default_repr = object.__repr__(self)
    #     class_name = type(self).__name__
    #     return f"{class_name}{default_repr}"
    def __repr__(self):
        class_name = type(self).__name__
        memory_address = hex(id(self))
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
        }
        # Update the dictionary with the __dict__ of Rectangle2D
        output_dict.update(self.__dict__)
        return output_dict

if __name__ == '__main__':
    # ça marche --> voici deux examples de shapes
    test = Rectangle2D(10, 0, 100, 100)
    print(test)
    # print(test.get_P1())

    test2 = Rectangle2D(QRectF(10, 0, 100, 100))
    print('pt2', test2)

    # then use my set_P1
    print(test2.setTopLeft(10,20)) # setTopLeft really sucks because set top left crops the rect for unknown reason --> use moveTopLeft instead --> ça marche maintenant comme je pensais
    print(test2)
    sys.exit(0)

    test.incompressible_width = 10
    test.incompressible_height = 5
    print(test.getIncompressibleWidth(),test.getIncompressibleHeight()) # --> that works --> this is a very easy way to get it then -−> this could also be borders around the image to fake it has the same size as another bigger image even if not --> can also be used by the tool that generates the stuff
    # in the case of rows that should be the space between cols or rows and in the case of panels it is both

    rect = QRectF(10, 0, 125, 256)
    print(rect.x())
    print(rect.y())
    print(rect.width())
    print(rect.height())

    rect.setTopLeft(QPointF(10.,20.))

    print('sum',rect.x()+rect.width(), type(rect.x()+rect.width())) # float

    print(rect.topLeft()) # this is same as get_P1 --> maybe stick to that

    rect.translate(10,20) # ça marche
    print(rect)

    print(test)
    test.set_to_scale(0.5) #scale is not applied to the object so its size isn't the real size
    # pb is that if I modify this class there will be tremendous repercusion on other classes --> be careful...
    print('scaled test',test)
    print('scaled test',test.boundingRect())
    print('scaled test',test.boundingRect(scaled=True))

    (test.x(), test.y(), test.width(), test.height())
    print(test.contains(QPointF(50, 50)))
    print(test.contains(QPointF(-1, -1)))
    print(test.contains(QPointF(0, 0)))
    print(test.contains(QPointF(100, 100)))
    print(test.contains(QPointF(100, 100.1)))


    # shall I do rotation


    # ideally plot it so that I can test all


    # p1 = test.p1()
    # print(p1.x(), p1.y())
    # p2 = test.p2()
    # print(p2.x(), p2.y())
    # print(test.arrow)
    # print(test.length()) # sqrt 2 --> 141
    # # if it's an arrow I can add easily all the stuff I need
    #
    # test = Rect2D(0, 0, 1, 1)
    # p1 = test.p1()
    # print(p1.x(), p1.y())
    # p2 = test.p2()
    # print(p2.x(), p2.y())
    # print(test.arrow)
    # import math
    # print(test.length() == math.sqrt(2))  # sqrt 2
    #
    # test2 = Rect2D()
    # p1 = test2.p1()
    # print(p1.x(), p1.y())
    # p2 = test2.p2()
    # print(p2.x(), p2.y())
    # print(test2.arrow)
