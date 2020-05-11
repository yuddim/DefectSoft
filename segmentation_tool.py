#Загрузка необходимых библиотек
import sys
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QMainWindow
from PyQt5.QtWidgets import QRadioButton, QGroupBox, QFileDialog, QLabel, QSlider, QLineEdit, QProgressBar
from PyQt5.QtGui  import QPixmap, QImage, QPainter, QColor, QCursor
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, QObject, QThread
import numpy as np
import cv2
from skimage.measure import label, regionprops, find_contours
import PIL
from fpdf import FPDF, HTMLMixin
from segmentation_model import SegmentationModel
from distutils.dir_util import copy_tree

obj_names = [
    '1 - Впадины',
    '2 - Вздутие',
    '3 - Складка',
    '4 - Заплатки',
    '5 - Отслаивание ковра',
    '6 - Разрывы ковра',
    '7 - Отсутствие ковра',
    '8 - Грибок/мох',
    '9 - Растрескивание']
obj_palete = [
    (0,0,128),
    (0,128,0),
    (255,165,0),
    (255,192,203),
    (255,0,0),
    (179,157,219),
    (192,202,51),
    (3,155,229),
    (121,85,72)]

# class LongRunning(QThread):
#     """
#     This class is not required if you're using the builtin
#     version of threading.
#     """
#     def __init__(self, form):
#         super().__init__()
#         self.form = form
#
#     def run(self):
#         """This overrides a default run function."""
#         #self.form.saveButtonClicked()
#         self.form.saveButtonEvent()


class CustomPDF(FPDF):
    def footer(self):
        self.set_y(-10)

        self.add_font('ArialU', '', 'C:/Windows/Fonts/Arial.ttf', uni=True)
        self.set_font('ArialU', '', 9)

        # Добавляем номер страницы
        page = str(self.page_no())
        self.cell(0, 10, page, 0, 0, 'C')

#Наследуем от QMainWindow:
class SegmentationTool(QMainWindow):
    #Переопределяем конструктор класса:
    def __init__(self, master=None):
        QMainWindow.__init__(self, master)
        self.flag = False
        self.img0 = []
        self.mask_inv = []
        self.initUI()
        self.model = SegmentationModel()


    #Создание макета проекта:
    def initUI(self):
        self.file_mk_dir = ""
        self.file_dir = ""
        self.filenames = []
        self.file_index = 0
        self.show_markup = True
        self.condition = False
        self.leftclickedflag = False
        self.rightclickedflag = False

        self.mainwidget = QWidget(self)
        self.setCentralWidget(self.mainwidget)

        self.projectName = ""
        self.coefList = []
        self.reportList = []

        #Создание обозначения кнопок:
        self.SelectImDirButton = QPushButton("Выбрать проект...")
        self.leProjectName = QLineEdit("Объект -")
        self.leProjectName.returnPressed.connect(self.changeProjectName)
        self.RecognizeFCNButton = QPushButton("Распознать дефекты")
        self.LoadMarkupButton = QPushButton("Загрузить разметку...")
        self.SaveAllImButton = QPushButton("Сохранить разметку...")
        self.FindBlobsButton = QPushButton("Рассчитать области")
        self.SaveButton = QPushButton("Сгенерировать/сохранить отчет")
        self.GotoButton = QPushButton("К изображению:")

        self.RecognizeFCNButton.setEnabled(False)
        self.FindBlobsButton.setEnabled(False)
        self.LoadMarkupButton.setEnabled(False)
        self.SaveAllImButton.setEnabled(False)
        self.SaveButton.setEnabled(False)
        self.GotoButton.setEnabled(False)
        self.mode = 0

        self.le_img_id = QLineEdit()
        self.le_img_max = QLabel('/ -')

        self.LabelCoef = QLabel(' | Изображение - : мм in 1 pix:')
        self.le_Scale_Coef = QLineEdit('3.0')

        self.LabelRadius = QLabel("Радиус кисти, px('+','-'):") #Brush radius ('+','-'):
        self.value_Radius = QLineEdit('10')
        self.radius = int(self.value_Radius.text())

        self.imageLabel = QLabel()

        self.defaultImWidth = 1500
        self.defaultImHeight = 750


        self.imageLabel.setMinimumWidth(self.defaultImWidth)
        self.imageLabel.setMaximumWidth(self.defaultImWidth)
        self.imageLabel.setMinimumHeight(self.defaultImHeight)
        self.imageLabel.setMaximumHeight(self.defaultImHeight)
        #Связь кнопок с методами их исполнения:
        self.FindBlobsButton.clicked.connect(self.FindBlobsEasyButtonClicked)
        self.RecognizeFCNButton.clicked.connect(self.RecognizeFCNButtonClicked)
        self.LoadMarkupButton.clicked.connect(self.LoadMarkupButtonClicked)
        self.SelectImDirButton.clicked.connect(self.selectImDirButtonClicked)
        self.SaveAllImButton.clicked.connect(self.SaveAllImButtonClicked)
        #self.SelectMkDirButton.clicked.connect(self.selectMkDirButtonClicked)
        self.SaveButton.clicked.connect(self.saveButtonClicked)
        self.GotoButton.clicked.connect(self.GotoButtonClicked)

        self.hbox = QHBoxLayout()

        self.hbox.addWidget(self.SelectImDirButton)
        self.hbox.addWidget(self.leProjectName)

        self.hbox.addWidget(self.RecognizeFCNButton)
        self.hbox.addWidget(self.LoadMarkupButton)
        self.hbox.addWidget(self.SaveAllImButton)
        self.hbox.addWidget(self.FindBlobsButton)
        self.hbox.addWidget(self.SaveButton)
        self.hbox.addWidget(self.GotoButton)
        self.hbox.addWidget(self.le_img_id)
        self.hbox.addWidget(self.le_img_max)

        self.hbox.addWidget(self.LabelCoef)
        self.hbox.addWidget(self.le_Scale_Coef)

        self.vcontrolboxwidget = QGroupBox()

        self.vcontrolbox = QVBoxLayout()

        self.LeftImButton = QPushButton("< Пред. ('a')")
        self.RightImButton = QPushButton("> След. ('d')")
        self.SaveImButton = QPushButton(" Сохранить ('s')")
        self.HideButton = QPushButton(" Скрыть разметку ('w')")
        self.SLeftImButton = QPushButton("< Сохранить+Пред. ('q')")
        self.SRightImButton = QPushButton("> Сохранить+ След. ('e')")

        self.RightImButton.clicked.connect(self.NextButtonClicked)
        self.LeftImButton.clicked.connect(self.PrevButtonClicked)
        self.SaveImButton.clicked.connect(self.saveImButtonClicked)
        self.HideButton.clicked.connect(self.HideButtonClicked)
        self.SLeftImButton.clicked.connect(self.SLeftImButtonClicked)
        self.SRightImButton.clicked.connect(self.SRightImButtonClicked)

        self.vcontrolbox.addWidget(self.RightImButton)
        self.vcontrolbox.addWidget(self.LeftImButton)
        self.vcontrolbox.addWidget(self.SRightImButton)
        self.vcontrolbox.addWidget(self.SLeftImButton)
        self.vcontrolbox.addWidget(self.SaveImButton)
        self.vcontrolbox.addWidget(self.HideButton)

        self.brush_group = QGroupBox('Тип разметки') #'Brush type'
        self.vbrushbox = QVBoxLayout()
        self.radio_no_brush = QRadioButton('Автоматическая') #'No Brush'
        self.radio_brush_circle = QRadioButton('Круглая кисть') #'Circle'
        self.radio_brush_rect = QRadioButton('Квадратная кисть')  #'Rectangle'
        self.vbrushbox.addWidget(self.radio_no_brush)
        self.vbrushbox.addWidget(self.radio_brush_circle)
        self.vbrushbox.addWidget(self.radio_brush_rect)
        self.radio_no_brush.toggled.connect(self.rbtnbrush)
        self.radio_brush_circle.toggled.connect(self.rbtnbrush)
        self.radio_brush_rect.toggled.connect(self.rbtnbrush)
        self.radio_no_brush.setChecked(True)
        self.brush_type = 0
        self.vbrushbox.addWidget(self.LabelRadius)
        self.vbrushbox.addWidget(self.value_Radius)
        self.brush_group.setLayout(self.vbrushbox)
        self.vcontrolbox.addWidget(self.brush_group)

        self.radio_group = QGroupBox('Типы дефектов кровли')
        self.rb_array = []
        index = 0
        self.vgroupbox = QVBoxLayout()
        for name in obj_names:
            self.rb_array.append(QRadioButton(name))
            color_str = str(obj_palete[index])
            self.rb_array[index].setStyleSheet("color: rgb"+color_str) #background-
            self.vgroupbox.addWidget(self.rb_array[index])
            self.rb_array[index].toggled.connect(self.rbtnstate)
            index += 1
        self.obj_index = 0 #global object index
        self.rb_array[self.obj_index].setChecked(True)
        self.vgroupbox.addStretch(1)
        self.radio_group.setLayout(self.vgroupbox)

        self.vcontrolbox.addWidget(self.radio_group)

        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        self.vcontrolbox.addWidget(self.pbar)

        #Создание формы приложения:
        self.pixmap = QPixmap()

        self.mainhbox = QHBoxLayout()

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.imageLabel)
        #Создание начального вида окна приложения:
        self.setGeometry(300, 300, 700, 500)
        self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0')
        self.setWindowIcon(QtGui.QIcon('resources/logo.png'))

        self.mainhbox.addLayout(self.vbox)
        self.mainhbox.addLayout(self.vcontrolbox)

        self.mainwidget.setLayout(self.mainhbox)
        self.setFullMouseTracking(True)

    def changeProjectName(self):
        self.projectName = self.leProjectName.text()
        if len(self.filenames)>0:
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Файл: ' + self.filenames[0])
        else:
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Нет файлов')



    def setFullMouseTracking(self, flag):
        def recursive_set(parent):
            for child in parent.findChildren(QObject):
                try:
                    child.setMouseTracking(flag)
                except:
                    pass
                recursive_set(child)

        QWidget.setMouseTracking(self, flag)
        recursive_set(self)

    def check_and_repaint_cursor(self, e):
        condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
        self.repaint_cursor(sc)


    def repaint_cursor(self, scale_coef):
        if(not self.condition or self.radio_no_brush.isChecked()):
            self.draw_cursor_default()
        else:
            radius = int(int(self.value_Radius.text())*scale_coef)
            r = obj_palete[self.obj_index][0]
            g = obj_palete[self.obj_index][1]
            b = obj_palete[self.obj_index][2]
            if(self.radio_brush_circle.isChecked()):
                self.draw_cursor_circle(radius, (r, g, b, 128))
            else:
                self.draw_cursor_rectangle(radius, (r, g, b, 128))
        return

    def draw_cursor_circle(self, radius, color):
        diameter = 2*radius
        self.m_LPixmap = QPixmap(diameter,diameter)
        self.m_LPixmap.fill(Qt.transparent)
        self.painter = QPainter(self.m_LPixmap)
        self.brush_color = QColor(color[0], color[1], color[2], color[3])
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(self.brush_color)
        self.painter.drawEllipse(0,0,diameter,diameter)
        self.m_cursor = QCursor(self.m_LPixmap)
        self.setCursor(self.m_cursor)
        self.painter.end()
        return

    def draw_cursor_rectangle(self, radius, color):
        width = 2 * radius
        self.m_LPixmap = QPixmap(width, width)
        self.m_LPixmap.fill(Qt.transparent)
        self.painter = QPainter(self.m_LPixmap)
        self.brush_color = QColor(color[0], color[1], color[2], color[3])
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(self.brush_color)
        self.painter.drawRect(0, 0, width, width)
        self.m_cursor = QCursor(self.m_LPixmap)
        self.setCursor(self.m_cursor)
        self.painter.end()
        return

    def draw_cursor_default(self):
        m_cursor = QCursor()
        self.setCursor(m_cursor)
        return

    def rbtnstate(self):
        radiobutton = self.sender()

        if radiobutton.isChecked():
            index = 0
            for name in obj_names:
                if(radiobutton.text()==name):
                    self.obj_index = index
                index += 1
            self.statusBar().showMessage('Selected index ' + str(self.obj_index))


    def rbtnbrush(self):
        radiobutton = self.sender()
        if radiobutton.isChecked():
            if(radiobutton.text() == 'Автоматическая'):
                self.brush_type = 0
            elif (radiobutton.text() == 'Круглая кисть'):
                self.brush_type = 1
            elif (radiobutton.text() == 'Квадратная кисть'):
                self.brush_type = 2
            else:
                self.brush_type = 0
            self.statusBar().showMessage('Тип разметки ' + str(self.brush_type) +
                                         ' : ' + radiobutton.text()) #'Brush type '

    #Метод открытия изображения:
    def selectFileButtonClicked(self):
        self.image_file = QFileDialog.getOpenFileName(None, 'Открыть файл', 'c:/Computer_Vision/MultiClass_Segmentation_Tool/Test_samples',
                                    'JPG Files(*.jpg);; PNG Files(*.png)')[0]
        if(self.image_file != ''):
            self.load_image_file(self.image_file)

    #Метод выбора папки с изображениями:
    def selectImDirButtonClicked(self):
        self.prj_dir = str(QFileDialog.getExistingDirectory(self, "Select directory with images", "projects"))
        if(self.prj_dir !=''):
            self.projectName = os.path.basename(self.prj_dir)
            self.leProjectName.setText(self.projectName)
            self.file_dir  = self.prj_dir + '/Фотоматериалы'
            self.file_mk_dir = self.prj_dir + '/Разметка'
            os.makedirs(self.file_mk_dir, exist_ok=True)
            self.file_report_dir = self.prj_dir + '/Отчеты'
            os.makedirs(self.file_report_dir, exist_ok=True)

            self.white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
            self.filenames = []

            for filename in sorted(os.listdir(self.file_dir+'')):
                is_valid = False
                for extension in self.white_list_formats:
                    if filename.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.filenames.append(filename)
                    self.coefList.append(1.0)

            if(len(self.filenames)==0):
                return

            self.le_img_max.setText('/ '+str(len(self.filenames)))

            self.image_file = self.file_dir + '/'+ self.filenames[0]
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0'+
                                ' | Проект : '+self.projectName+' | File: ' + self.filenames[0])
            self.le_img_id.setText(str(1))

            self.load_image_file(self.image_file)
            self.LabelCoef.setText(' | Изображение - ' + str(self.file_index+1) + ' : мм in 1 pix:')

            message = "Images directory is successfully selected: " + self.file_dir

            self.RecognizeFCNButton.setEnabled(True)
            self.LoadMarkupButton.setEnabled(True)
            self.SaveAllImButton.setEnabled(False)
            self.FindBlobsButton.setEnabled(False)
            self.SaveButton.setEnabled(False)
            self.GotoButton.setEnabled(True)

            self.mode = 1

            self.statusBar().showMessage(message)

    def RecognizeFCNButtonClicked(self):

        if (len(self.filenames) == 0):
            return
        self.statusBar().showMessage("Processing start..")
        self.repaint()
        self.model.run_model_on_folder(self.file_dir, self.file_mk_dir)

        self.load_image_file(file_name=self.image_file, isMask=True)

        message = "Images are successfully recognized: " + self.file_dir

        self.RecognizeFCNButton.setEnabled(True)
        self.FindBlobsButton.setEnabled(True)
        self.LoadMarkupButton.setEnabled(True)
        self.SaveAllImButton.setEnabled(True)
        self.SaveButton.setEnabled(False)
        self.GotoButton.setEnabled(True)

        self.mode = 2

        self.statusBar().showMessage(message)

    def LoadMarkupButtonClicked(self):
        if (len(self.filenames) == 0):
            return

        self.file_mk_dir = str(
            QFileDialog.getExistingDirectory(self, "Select Markup Directory", "", QFileDialog.ShowDirsOnly))
        message = "Markup directory is successfully selected: " + self.file_mk_dir
        self.statusBar().showMessage(message)

        self.statusBar().showMessage("Loading start..")

        self.load_image_file(file_name=self.image_file, isMask=True)

        message = "Images are successfully recognized: " + self.file_dir

        self.RecognizeFCNButton.setEnabled(True)
        self.FindBlobsButton.setEnabled(True)
        self.LoadMarkupButton.setEnabled(True)
        self.SaveAllImButton.setEnabled(True)
        self.SaveButton.setEnabled(False)
        self.GotoButton.setEnabled(True)

        self.mode = 2

        self.statusBar().showMessage(message)

    def selectMkDirButtonClicked(self):
        self.file_mk_dir = str(QFileDialog.getExistingDirectory(self, "Select Markup Directory", "D:/Datasets/Students_monitoring"))
        message = "Markup directory is successfully selected: " + self.file_mk_dir
        self.statusBar().showMessage(message)


    #Метод обработки изображения с помощью HSV:
    def startHSVButtonClicked(self):
        if self.pixmap.isNull() != True:
            self.img0 = cv2.imread(self.image_file, 1)
            self.img = cv2.GaussianBlur(self.img0, (5, 5), 5)
            #Перевод изображения из BGR в цветовую систему HSV:
            self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            #Задание диапозона поиска цветовых пятен:
            self.lower_range = np.array([self.HueMin, self.SaturationMin, self.ValueMin], dtype=np.uint8)
            self.upper_range = np.array([self.HueMax, self.SaturationMax, self.ValueMax], dtype=np.uint8)
            #Маска изображения, выделяющая пятно:
            self.mask = cv2.inRange(self.hsv, self.lower_range, self.upper_range)
            self.mask_inv = self.binary_to_color_with_pallete(self.mask, obj_palete[self.obj_index])

            #Наложение инвертированной маски на изображение:
            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)

            self.print_image_on_label(self.rez, self.imageLabel)
            self.flag = True
        else:
            pass

    def CheckNumObjectsPath(self):
        self.file_index = 0
        ind = 0
        output_file = open('statistics\\folder_statistics_ethalon.txt', 'w')
        out_text = "n_obj\tfilename\n"
        output_file.write(out_text)

        for ind in range(len(self.filenames)):
            n_objects = self.FindBlobsEasyButtonClicked()
            out_text = "%d\t%s\n" % (n_objects, self.filenames[self.file_index])
            output_file.write(out_text)
            self.NextButtonClicked()
        output_file.close()

    def FindBlobsEasyButtonClicked(self):
        if (self.pixmap.isNull() != True and len(self.mask_inv) != 0):


            # Цвет текста координат:
            self.box_color = (255, 255, 255)
            self.shadow_color = (0, 0, 0)

            self.coefList[self.file_index] = float(self.le_Scale_Coef.text())

            self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            # label image regions
            for objIndex, objName in enumerate(obj_names):
                rgbPallete = obj_palete[objIndex]
                bgrPallete = [rgbPallete[2], rgbPallete[1], rgbPallete[0]]
                self.mask = (self.mask_inv == bgrPallete)
                label_image = label(self.mask[:, :, 0] * self.mask[:, :, 1] * self.mask[:, :, 2])

                ind = 1
                for region in regionprops(label_image):
                    # skip small images
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox

                    S = (maxc - minc) * (maxr - minr)

                    if (S < 200):
                        continue

                    coords = region.coords
                    coords = coords[:, ::-1]
                    filler = cv2.convexHull(coords)

                    cv2.polylines(self.rez, [filler], True, self.box_color, 3)
                    scaleCoef = self.coefList[self.file_index]

                    maskAreaScaled = region.area * (scaleCoef * scaleCoef) / 1000000 # в м. кв.
                    convexHullAreaScaled = cv2.contourArea(filler) * (scaleCoef * scaleCoef) / 1000000 # в м. кв.
                    boxAreaScaled = S * (scaleCoef * scaleCoef) / 1000000 # в м. кв.

                    maskArea = region.area
                    convexHullArea = cv2.contourArea(filler)
                    boxArea = S

                    imageArea = self.img0.shape[0] * self.img0.shape[1]
                    maskAreaRate = region.area / imageArea * 100 # в % от площади изображения.
                    convexHullAreaRate = cv2.contourArea(filler) / imageArea  * 100 # в % от площади изображения.
                    boxAreaRate = S / imageArea  * 100 # в % от площади изображения.

                    cv2.rectangle(self.rez, (minc+1, minr+1), (maxc-1, maxr-1), self.shadow_color, 1)
                    cv2.rectangle(self.rez, (minc, minr), (maxc, maxr), self.box_color, 1)

                    cv2.putText(self.rez, "%d:%d" % (objIndex + 1, ind),
                                (int(minc + 3), int(minr + 22)), cv2.FONT_HERSHEY_PLAIN, 1.4, self.shadow_color, 3)
                    cv2.putText(self.rez, "%d:%d" % (objIndex + 1, ind),
                                (int(minc + 3), int(minr + 22)), cv2.FONT_HERSHEY_PLAIN, 1.4, self.box_color, 1)
                    ind += 1

            self.print_image_on_label(self.rez, self.imageLabel)
            # self.reportList.append(imageReport)

            self.RecognizeFCNButton.setEnabled(True)
            self.FindBlobsButton.setEnabled(True)
            self.LoadMarkupButton.setEnabled(True)
            self.SaveAllImButton.setEnabled(True)
            self.SaveButton.setEnabled(True)
            self.GotoButton.setEnabled(True)

            self.mode = 3
            return ind
        else:
            return 0
    # Метод, в котором изменяется значение отдельных ползунков и связывается с цветом со строкой ввода диапазона:
    def valueChange(self):
        self.HueMin = self.slider_HueMin.value()
        self.value_line_1.setText(str(self.HueMin))
        self.HueMax = self.slider_HueMax.value()
        self.value_line_2.setText(str(self.HueMax))
        self.SaturationMin = self.slider_SaturationMin.value()
        self.value_line_3.setText(str(self.SaturationMin))
        self.SaturationMax = self.slider_SaturationMax.value()
        self.value_line_4.setText(str(self.SaturationMax))
        self.ValueMin = self.slider_ValueMin.value()
        self.value_line_5.setText(str(self.ValueMin))
        self.ValueMax = self.slider_ValueMax.value()
        self.value_line_6.setText(str(self.ValueMax))

    def valueChangePress(self):
        #Изменяет значение диапазона цвета на введённое в строку ввода
        old = self.Coef
        # Считывается новое значение
        new = self.le_Scale_Coef.text()
        # Выполняется проверка на корректность введённого значения
        rez = self.chekForSymb(old, new)
        # Присваивается более коректное значение
        self.Coef = rez
        # Выводится значение в строку ввода
        self.le_Scale_Coef.setText(str(rez))

        # Запоминается прежнее значение на случай ввода недопустимого значения
        old = self.HueMin
        # Считывается новое значение
        new = self.value_line_1.text()
        # Выполняется проверка на корректность введённого значения
        rez = self.chekForSymb(old, new)
        # Присваивается более коректное значение
        self.HueMin = rez
        # Ползунок устанавливается на новое или старое значение
        self.slider_HueMin.setValue(rez)
        # Выводится значение в строку ввода
        self.value_line_1.setText(str(rez))

        old = self.HueMax
        new = self.value_line_2.text()
        rez = self.chekForSymb(old, new)
        self.HueMax = rez
        self.slider_HueMax.setValue(rez)
        self.value_line_2.setText(str(rez))

        old = self.SaturationMin
        new = self.value_line_3.text()
        rez = self.chekForSymb(old, new)
        self.SaturationMin = rez
        self.slider_SaturationMin.setValue(rez)
        self.value_line_3.setText(str(rez))

        old = self.SaturationMax
        new = self.value_line_4.text()
        rez = self.chekForSymb(old, new)
        self.SaturationMax = rez
        self.slider_SaturationMax.setValue(rez)
        self.value_line_4.setText(str(rez))

        old = self.ValueMin
        new = self.value_line_5.text()
        rez = self.chekForSymb(old, new)
        self.ValueMin = rez
        self.slider_ValueMin.setValue(rez)
        self.value_line_5.setText(str(rez))

        old = self.ValueMax
        new = self.value_line_6.text()
        rez = self.chekForSymb(old, new)
        self.ValueMax = rez
        self.slider_ValueMax.setValue(rez)
        self.value_line_6.setText(str(rez))

    def chekForSymb(self, old, new):
        #Возвращает число new в случае его корректности или же old в случае некорректности числа new
        # Запускается генератор множества
        consts = {str(i) for i in range(10)}
        # Перебираются все элементы в new
        for el in new:
            # Если число содержит символы, не являющиеся цифрами, завершается проверка числа на корректность
            if el not in consts:
                # На выход функции подаётся прежнее значение, и функция завершается
                return old
                exit()
        if int(new) < 255:
            return int(new)
        else:
            return 255
    def generateBlobsAllImages(self):
        self.reportList = []
        self.pbar.setVisible(True)
        if (self.pixmap.isNull() != True and len(self.mask_inv) != 0):
            stepCoef = 100/len(self.filenames)
            for fileIndex in range(len(self.filenames)):
                self.pbar.setValue(int(fileIndex*stepCoef))
                self.file_index = fileIndex
                self.image_file = self.file_dir + '/'+ self.filenames[self.file_index]
                # Цвет текста координат:
                self.box_color = (255, 255, 255)
                self.shadow_color = (0, 0, 0)

                self.coefList[fileIndex] = float(self.le_Scale_Coef.text())

                imageReport = []
                imageReport.append(self.image_file)

                res_fname, res_extension = os.path.splitext(self.image_file)
                res_fname = os.path.basename(res_fname)
                target_fname_mask = self.file_report_dir + '/Результат распознавания/' + res_fname + res_extension

                self.load_image_file(self.image_file, isMask=True)

                imageReport.append(target_fname_mask)
                imageReport.append(self.coefList[self.file_index])
                imageReport.append(self.img0.shape)

                self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
                # label image regions
                for objIndex, objName in enumerate(obj_names):
                    rgbPallete = obj_palete[objIndex]
                    bgrPallete = [rgbPallete[2], rgbPallete[1], rgbPallete[0]]
                    self.mask = (self.mask_inv == bgrPallete)
                    label_image = label(self.mask[:, :, 0] * self.mask[:, :, 1] * self.mask[:, :, 2])

                    ind = 1

                    for region in regionprops(label_image):
                        # skip small images
                        # draw rectangle around segmented coins
                        minr, minc, maxr, maxc = region.bbox

                        S = (maxc - minc) * (maxr - minr)

                        if (S < 200):
                            continue

                        coords = region.coords
                        coords = coords[:, ::-1]
                        filler = cv2.convexHull(coords)

                        cv2.polylines(self.rez, [filler], True, self.box_color, 3)

                        scaleCoef = self.coefList[self.file_index]

                        maskAreaScaled = region.area * (scaleCoef * scaleCoef) / 1000000 # в м. кв.
                        convexHullAreaScaled = cv2.contourArea(filler) * (scaleCoef * scaleCoef) / 1000000 # в м. кв.
                        boxAreaScaled = S * (scaleCoef * scaleCoef) / 1000000 # в м. кв.

                        maskArea = region.area
                        convexHullArea = cv2.contourArea(filler)
                        boxArea = S

                        imageArea = self.img0.shape[0] * self.img0.shape[1]
                        maskAreaRate = region.area / imageArea * 100 # в % от площади изображения.
                        convexHullAreaRate = cv2.contourArea(filler) / imageArea  * 100 # в % от площади изображения.
                        boxAreaRate = S / imageArea  * 100 # в % от площади изображения.

                        imageReport.append(["%d:%d" % (objIndex + 1, ind), objName,
                                            maskArea, convexHullArea, boxArea,
                                            maskAreaRate, convexHullAreaRate, boxAreaRate,
                                            maskAreaScaled, convexHullAreaScaled, boxAreaScaled])

                        cv2.rectangle(self.rez, (minc+1, minr+1), (maxc-1, maxr-1), self.shadow_color, 1)
                        cv2.rectangle(self.rez, (minc, minr), (maxc, maxr), self.box_color, 1)

                        cv2.putText(self.rez, "%d:%d" % (objIndex + 1, ind),
                                    (int(minc + 3), int(minr + 22)), cv2.FONT_HERSHEY_PLAIN, 1.4, self.shadow_color, 3)
                        cv2.putText(self.rez, "%d:%d" % (objIndex + 1, ind),
                                    (int(minc + 3), int(minr + 22)), cv2.FONT_HERSHEY_PLAIN, 1.4, self.box_color, 1)
                        ind += 1

                    res_fname, res_extension = os.path.splitext(self.image_file)
                    res_fname = os.path.basename(res_fname)

                    os.makedirs(self.file_report_dir + '/Маски', exist_ok=True)
                    os.makedirs(self.file_report_dir + '/Результат распознавания', exist_ok=True)
                    target_fname_mask = self.file_report_dir + '/Результат распознавания/' + res_fname + res_extension

                    target_fname_markup = self.file_report_dir + '/Маски/' + res_fname + '.png'

                    self.imwrite_utf8(target_fname_markup, self.mask_inv)
                    self.imwrite_utf8(target_fname_mask, self.rez)

                self.reportList.append(imageReport)

            self.le_Scale_Coef.setText(str(self.coefList[self.file_index]))
            self.LabelCoef.setText(' | Изображение - ' + str(self.file_index+1) + ' : мм in 1 pix:')
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Файл: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index+1))
            self.pbar.setValue(100)
            self.FindBlobsEasyButtonClicked()
            self.pbar.setVisible(False)
            return ind
        else:
            return 0

    # Метод сохранения отчета:
    def saveButtonClicked(self):
        self.saveButtonEvent()

    def saveButtonEvent(self):
        if (self.flag):
            self.generateBlobsAllImages()

            pdf = CustomPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()

            pdf.add_font('ArialU', '', 'C:/Windows/Fonts/Arial.ttf', uni=True)
            pdf.set_font('ArialU', '', 14)

            # pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="ОТЧЕТ", ln=1, align="C")
            pdf.cell(200, 10, txt='об обследовании объекта', ln=1, align="C")

            pdf.cell(200, 10, txt=self.projectName, ln=1, align="C")
            pdf.set_line_width(0.5)
            pdf.set_draw_color(0, 0, 0)
            pdf.line(10, 40, 200, 40)
            pdf.cell(200, 3, txt='', ln=1, align="C")

            for fileId, imagePath in enumerate(self.filenames):
                fullImageFilePath = self.file_dir + '/' + imagePath

                pdf.image(fullImageFilePath, x=None, y=None, w=190)
                pdf.cell(200, 10, txt="{}".format(os.path.basename(fullImageFilePath)), ln=1)

            for imageReport in self.reportList:
                pdf.add_page()
                pdf.cell(200, 10, txt="РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ДЕФЕКТОВ", ln=1, align="C")
                image_file = imageReport[0]
                maskFileName = imageReport[1]
                #self.load_image_file(file_name=self.image_file, isMask=True)
                coef = imageReport[2]
                imgShape = imageReport[3]
                pdf.cell(200, 10, txt="Изображение : {}".format(os.path.basename(image_file)), ln=1)
                pdf.cell(200, 10, txt="Масштабный коэффициент (мм в 1 пикселе): {}".format(coef), ln=1)
                pdf.cell(200, 10, txt="Размер изображения: {0}x{1}".format(imgShape[0], imgShape[1]), ln=1)

                pdf.image(maskFileName, x=None, y=None, w=190)
                pdf.cell(200, 10, txt="Результат для изображения {}".format(os.path.basename(maskFileName)), ln=1)

                col_width = pdf.w / 6.5
                row_height = pdf.font_size
                spacing = 1

                pdf.cell(int(col_width/2.5), row_height * spacing, txt="Код", border=1)
                pdf.cell(col_width*2, row_height * spacing, txt="Тип дефекта кровли", border=1)
                pdf.cell(int(col_width*0.7), row_height * spacing, txt="Маска,%", border=1)
                pdf.cell(col_width, row_height * spacing, txt="Маска,кв.м", border=1)
                pdf.cell(int(col_width*0.9), row_height * spacing, txt="Область,%", border=1)
                pdf.cell(col_width, row_height * spacing, txt="Область,кв.м", border=1)

                pdf.ln(row_height * spacing)

                maskAreaRateSum = 0
                convexHullAreaRateSum = 0
                maskAreaScaledSum = 0
                convexHullAreaScaledSum = 0

                for idImg, imgInfo in enumerate(imageReport):
                    if idImg < 4:
                        continue
                    imgInfo = imageReport[idImg]
                    regionCode = imgInfo[0]
                    regionCatergory = imgInfo[1]
                    maskArea = imgInfo[2]
                    convexHullArea = imgInfo[3]
                    boxArea = imgInfo[4]
                    maskAreaRate = imgInfo[5]
                    convexHullAreaRate = imgInfo[6]
                    boxAreaRate = imgInfo[7]
                    maskAreaScaled = imgInfo[8]
                    convexHullAreaScaled = imgInfo[9]
                    boxAreaScaled = imgInfo[10]

                    maskAreaRateSum += maskAreaRate
                    if (maskAreaRateSum > 100):
                        maskAreaRateSum = 100.0
                    convexHullAreaRateSum += convexHullAreaRate
                    if (convexHullAreaRateSum > 100):
                        convexHullAreaRateSum = 100.0
                    maskAreaScaledSum += maskAreaScaled
                    convexHullAreaScaledSum += convexHullAreaScaled

                    pdf.cell(int(col_width/2.5), row_height * spacing,txt=str(regionCode), border=1)
                    pdf.cell(col_width*2, row_height * spacing, txt=str(regionCatergory), border=1)
                    pdf.cell(int(col_width*0.7), row_height * spacing, txt="%.4f"%(maskAreaRate), border=1)
                    pdf.cell(col_width, row_height * spacing, txt="%.6f"%(maskAreaScaled), border=1)
                    pdf.cell(int(col_width*0.9), row_height * spacing, txt="%.4f"%(convexHullAreaRate), border=1)
                    pdf.cell(col_width, row_height * spacing, txt="%.6f"%(convexHullAreaScaled), border=1)

                    pdf.ln(row_height * spacing)

                pdf.cell(int(col_width/2.5), row_height * spacing, txt="", border=1)
                pdf.cell(col_width * 2, row_height * spacing, txt="Итого: ", border=1)
                pdf.cell(int(col_width*0.7), row_height * spacing, txt="%.4f" % (maskAreaRateSum), border=1)
                pdf.cell(col_width, row_height * spacing, txt="%.6f" % (maskAreaScaledSum), border=1)
                pdf.cell(int(col_width*0.9), row_height * spacing, txt="%.4f" % (convexHullAreaRateSum), border=1)
                pdf.cell(col_width, row_height * spacing, txt="%.6f" % (convexHullAreaScaledSum), border=1)

                pdf.ln(row_height * spacing)


                message = "Отчет успешно сохранен в папке: " + self.file_report_dir

            pdf.add_page()
            pdf.cell(200, 30, txt="Отчет проверен и согласован ", ln=1)
            pdf.cell(200, 30, txt="Дата: ________________________", ln=1)
            pdf.cell(200, 30, txt="Подпись: _____________________/_________________", ln=1)
            pdf.cell(200, 30, txt="Подпись: _____________________/_________________", ln=1)

            pdf.output(self.file_report_dir + '/Отчет_' + self.projectName + '_.pdf')
        self.statusBar().showMessage(message)


    #Метод сохранения изображения:
    def saveImButtonClicked(self):
        if (self.flag):
            res_fname, res_extension = os.path.splitext(self.image_file)
            res_fname = os.path.basename(res_fname)
            target_dir = self.file_mk_dir

            os.makedirs(target_dir, exist_ok=True)
            target_fname_markup = target_dir + '/' + res_fname + '.png'
            self.imwrite_utf8(target_fname_markup, self.mask_inv)
            message = "Файл успешно сохранен: " + target_fname_markup
            self.statusBar().showMessage(message)

    # Метод сохранения папки с изображениями:
    def SaveAllImButtonClicked(self):
        if (self.flag):
            self.saveImButtonClicked()
            dir = QFileDialog.getExistingDirectory(self,'Сохранить как', '', QFileDialog.ShowDirsOnly)
            if dir != '':
                os.makedirs(dir, exist_ok=True)
                copy_tree(self.file_mk_dir, dir)
                self.file_mk_dir = dir

                message = "Разметка успешно сохранена: " + self.file_mk_dir
                self.statusBar().showMessage(message)

    #Метод присвоения левой кнопке мыши метода рисования:
    def mousePressEvent(self, e):
        message = ""
        if e.button() == Qt.LeftButton:
            self.leftclickedflag = True
            self.rightclickedflag = False
            message = "Left click"
        if e.button() == Qt.RightButton:
            self.leftclickedflag = False
            self.rightclickedflag = True
            message = "Right click"
        self.statusBar().showMessage(message)

    def mouseReleaseEvent(self, e):
        message = ""
        if e.button() == Qt.LeftButton:
            self.leftclickedflag = False
            message = "Left click release"
        if e.button() == Qt.RightButton:
            self.rightclickedflag = False
            message = "Right click release"
        self.statusBar().showMessage(message)

    def grayscale_to_color(self, grayscale_im):
        im_reshaped = grayscale_im.reshape((grayscale_im.shape[0], grayscale_im.shape[1], 1))
        im_out = np.append(im_reshaped, im_reshaped,axis=2)
        im_out = np.append(im_out, im_reshaped, axis=2)
        return im_out
    def binary_to_color_with_pallete(self, binary_im, pallete_color):
        im_reshaped = binary_im.reshape((binary_im.shape[0], binary_im.shape[1], 1))/255
        im_out = np.append(im_reshaped*(255-pallete_color[0]), im_reshaped*(255-pallete_color[1]),axis=2)
        im_out = np.append(im_out, im_reshaped*(255-pallete_color[2]), axis=2)
        im_out = 255 - im_out
        return im_out

    #Метод рисования дефектов:
    def mouseMoveEvent(self, e):
        if self.brush_type == 0:
            return
        coords = e.pos()
        old_condition = self.condition
        self.condition = self.is_in_field(e, self.imageLabel)
        self.statusBar().showMessage(str(coords.x()) + ' ' + str(coords.y())+' '+str(self.condition))
        if (old_condition != self.condition and self.flag):
            self.repaint_cursor(self.sc)

        if self.leftclickedflag:
            condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if(condition):
                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, obj_palete[self.obj_index], int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0*(self.mask_inv/255)).astype(np.uint8)
                #self.rez = cv2.bitwise_and(self.img0, self.img0, mask=self.mask_inv)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Left move: True"
            else:
                message = "Left move: False"

            self.statusBar().showMessage(message)

        if self.rightclickedflag:
            condition, w, h, sc = self.check_paint_field(e, self.flag, self.img0, self.imageLabel)
            if (condition):
                self.mask_inv = self.draw_ellipse_on_mask(e, self.mask_inv, w, h, (255,255,255),
                                                          int(self.value_Radius.text()), self.brush_type)
                self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
                self.print_image_on_label(self.rez, self.imageLabel)
                message = "Right move: True"
            else:
                message = "Right move: False"
            self.statusBar().showMessage(message)
        e.accept()

    #Метод создания кисти для рисования:
    def ellips(self,e):
        self.paint = QPainter(self.qimg)
        self.paint.setBrush(QColor('black'))
        coords = e.pos()
        geom = self.imageLabel.geometry()
        coords_ed = coords - QPoint(geom.x(), geom.y())
        self.paint.drawEllipse(coords_ed, 10,10)
        self.update()

    # Метод создания кисти для рисования:
    def draw_ellipse_on_mask(self, e, mask_img, label_real_w, label_real_h, draw_color, radius, brush_type):
        coords = e.pos()
        geom = self.imageLabel.geometry()
        coords_ed = coords - QPoint(geom.x(), geom.y())

        mask_height = mask_img.shape[0]
        mask_width = mask_img.shape[1]
        pixmap_height = label_real_h
        pixmap_width = label_real_w

        real_x = int(coords_ed.x()*mask_width/pixmap_width)
        real_y = int(coords_ed.y() * mask_height / pixmap_height)

        draw_color_bgr = (draw_color[2], draw_color[1], draw_color[0])
        if (brush_type == 1):
            cv2.circle(mask_img, (real_x, real_y), radius, draw_color_bgr, -1)
        else:
            cv2.rectangle(mask_img, (real_x-radius, real_y-radius), (real_x+radius, real_y+radius), draw_color_bgr, -1)
        return mask_img

    def print_image_on_label(self, img, label_widget):
        height, width, channel = img.shape
        bytesPerLine = channel * width
        rgb_cvimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb_cvimage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.pixmap = QPixmap(q_img)

        w_p = self.pixmap.width()
        h_p = self.pixmap.height()

        if (w_p > h_p):
            label_new_width = self.defaultImWidth
            label_new_height = h_p * self.defaultImWidth / w_p
        else:
            label_new_width = w_p * self.defaultImHeight / h_p
            label_new_height = self.defaultImHeight

        label_widget.setMinimumWidth(label_new_width)
        label_widget.setMaximumWidth(label_new_width)
        label_widget.setMinimumHeight(label_new_height)
        label_widget.setMaximumHeight(label_new_height)

        # Вычисляем ширину окна изображения
        w = label_widget.width()
        # Вычисляем высоту окна изображения
        h = label_widget.height()
        self.pixmap = self.pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(self.pixmap)

        return True

    def check_paint_field(self, e, rec_flag, img, label_widget):
        if(rec_flag):
            img_height, img_width, img_channel = img.shape
            geom = label_widget.geometry()

            scale_x_w = geom.width()
            scale_y_h = geom.height()

            scale_ratio_img = img_height/img_width
            scale_ratio_label = geom.height() / geom.width()
            if(scale_ratio_img > scale_ratio_label):
                scale_x_w = geom.height()/scale_ratio_img
                scale_coef = geom.height() / img_height
            if (scale_ratio_img <= scale_ratio_label):
                scale_y_h = geom.width() * scale_ratio_img
                scale_coef = geom.width() / img_width
            coords = e.pos()

            x_max = geom.x() + scale_x_w
            y_max = geom.y() + scale_y_h
            condition_label = coords.x() > geom.x() and coords.x() < x_max and coords.y() > geom.y() and coords.y() < y_max
            if(condition_label):
                return True, scale_x_w, scale_y_h, scale_coef

        return False, 0, 0, 0

    def is_in_field(self, e, label_widget):
        geom = label_widget.geometry()

        coords = e.pos()

        x_max = geom.x() + geom.width()
        y_max = geom.y() + geom.height()
        condition_label = coords.x() > geom.x() and coords.x() < x_max and coords.y() > geom.y() and coords.y() < y_max
        if(condition_label):
            return True
        return False

    def calc_scale_coef(self, img, label_widget):
        img_height, img_width, img_channel = img.shape
        scale_coef = 1
        geom = label_widget.geometry()

        if(img_width!=0 and img_height!=0):
            scale_ratio_img = img_height/img_width
            scale_ratio_label = geom.height() / geom.width()
            if(scale_ratio_img > scale_ratio_label):
                scale_coef = geom.height() / img_height
            if (scale_ratio_img <= scale_ratio_label):
                scale_coef = geom.width() / img_width

        return scale_coef


    def NextButtonClicked(self):
        if(len(self.filenames)>self.file_index+1):
            self.coefList[self.file_index] = float(self.le_Scale_Coef.text())
            self.file_index += 1
            self.le_Scale_Coef.setText(str(self.coefList[self.file_index]))
            self.LabelCoef.setText(' | Изображение - ' + str(self.file_index+1)+ ' : мм in 1 pix:')
            self.image_file = self.file_dir + '/'+self.filenames[self.file_index]
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Файл: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index+1))
            if self.mode > 1:
                self.load_image_file(file_name=self.image_file, isMask=True)
                if self.mode == 3:
                    self.FindBlobsEasyButtonClicked()
            else:
                self.load_image_file(file_name=self.image_file)
            self.statusBar().showMessage('> next image')
        else:
            self.statusBar().showMessage('no next image')

    def PrevButtonClicked(self):
        if (self.file_index > 0):
            self.coefList[self.file_index] = float(self.le_Scale_Coef.text())
            self.file_index -= 1
            self.le_Scale_Coef.setText(str(self.coefList[self.file_index]))
            self.image_file = self.file_dir + '/' + self.filenames[self.file_index]
            self.LabelCoef.setText(' | Изображение - ' + str(self.file_index+1) + ' : мм in 1 pix:')
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Файл: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index+1))
            if self.mode > 1:
                self.load_image_file(file_name=self.image_file, isMask=True)
                if self.mode == 3:
                    self.FindBlobsEasyButtonClicked()
            else:
                self.load_image_file(file_name=self.image_file)
            self.statusBar().showMessage('< previous image')
        else:
            self.statusBar().showMessage('no previous image')

    def GotoButtonClicked(self):
        self.file_index = int(self.le_img_id.text())-1
        if (self.file_index >= 0 and len(self.filenames)>self.file_index):
            self.image_file = self.file_dir + '/' + self.filenames[self.file_index]
            self.setWindowTitle('Программа распознавания дефектов кровли - DefectSoft v.1.0' +
                                ' | Проект : ' + self.projectName + ' | Файл: ' + self.filenames[self.file_index])
            self.le_img_id.setText(str(self.file_index+1))
            if self.mode > 1:
                self.load_image_file(file_name=self.image_file, isMask=True)
                if self.mode == 3:
                    self.FindBlobsEasyButtonClicked()
            else:
                self.load_image_file(file_name=self.image_file)
            self.statusBar().showMessage('find image')
        else:
            self.statusBar().showMessage('no image')

    def imread_utf8(self, filename):
        try:
            pil_image = PIL.Image.open(filename).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            return open_cv_image
        except Exception as e:
            print(e)
            return None

    def imwrite_utf8(self, filename, img):
        try:
            cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_im = PIL.Image.fromarray(cv2_im)
            pil_im.save(filename)
            return True
        except Exception as e:
            print(e)
            return False
    #Загрузка изображения на форму
    def load_image_file(self, file_name, isMask=False, path_mask = None):
        self.img0 = self.imread_utf8(file_name)

        if (type(self.img0) is np.ndarray):
            if isMask:
                res_fname, res_extension = os.path.splitext(self.image_file)
                res_fname = os.path.basename(res_fname)
                if path_mask is None:
                    mask_path = self.file_mk_dir+'/' + res_fname+'.png'
                else:
                    mask_path = path_mask + '/' + res_fname + '.png'
                self.mask_inv = self.imread_utf8(mask_path)
                if(type(self.mask_inv) is np.ndarray):
                    self.rez = (self.img0 * (self.mask_inv / 255)).astype(np.uint8)
            else:
                self.rez = self.img0
                self.mask_inv = np.ones(self.img0.shape, dtype='uint8')*255
            self.sc = self.calc_scale_coef(self.img0, self.imageLabel)
            self.flag = True
            self.print_image_on_label(self.rez, self.imageLabel)
        else:
            self.flag = False

    # Загрузка изображения на форму
    def load_just_image_file(self, file_name):
        self.img0 = self.imread_utf8(file_name)
        if (type(self.img0) is np.ndarray):
            self.rez = self.img0
            self.print_image_on_label(self.rez, self.imageLabel)
            self.sc = self.calc_scale_coef(self.img0, self.imageLabel)
            self.flag = True
        else:
            self.flag = False

    def HideButtonClicked(self):
        if not self.flag:
            return

        if self.show_markup:
            self.show_markup = False
            self.flag = False
            self.load_just_image_file(self.image_file)
            self.HideButton.setText(" Показать разметку ('w')")
        else:
            self.show_markup = True
            self.flag = True
            self.HideButton.setText(" Скрыть разметку ('w')")
            if self.mode > 1:
                self.load_image_file(self.image_file, isMask=True)
                if self.mode == 3:
                    self.FindBlobsEasyButtonClicked()

    def SLeftImButtonClicked(self):
        self.saveImButtonClicked()
        self.PrevButtonClicked()

    def SRightImButtonClicked(self):
        self.saveImButtonClicked()
        self.NextButtonClicked()

    def IncreaseRadius(self, step):
        if(self.radius < 1000 and self.flag and self.brush_type > 0):
            self.radius += step
            self.value_Radius.setText(str(self.radius))
            self.repaint_cursor(self.sc)

    def DerceaseRadius(self, step):
        if(self.radius > step and self.flag and self.brush_type > 0):
            self.radius -= step
            self.value_Radius.setText(str(self.radius))
            self.repaint_cursor(self.sc)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A:
            self.PrevButtonClicked()
        elif e.key() == Qt.Key_D:
            self.NextButtonClicked()
        elif e.key() == Qt.Key_S:
           self.saveButtonClicked()
        elif e.key() == Qt.Key_W:
            self.HideButtonClicked()
        elif e.key() == Qt.Key_Q:
           self.SLeftImButtonClicked()
        elif e.key() == Qt.Key_E:
           self.SRightImButtonClicked()
        elif e.key() == Qt.Key_Plus:
           self.IncreaseRadius(1)
        elif e.key() == Qt.Key_Minus:
           self.DerceaseRadius(1)

    def wheelEvent(self, event):
        numDegrees = event.angleDelta().y() / 8
        numSteps = numDegrees / 15.0
        if(numSteps > 0):
            self.IncreaseRadius(2)
        elif (numSteps < 0):
            self.DerceaseRadius(2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SegmentationTool()
    ex.move(0, 0)
    ex.show()

    sys.exit(app.exec_())