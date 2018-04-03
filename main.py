import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import psutil
import time
import pandas
import matplotlib
import numpy as np
import keras
import glob
import pandas
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib import pyplot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import proj3d
from matplotlib.figure import Figure
from PIL import Image
from sklearn.manifold import TSNE
from pyqtwindow import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QFileSystemModel
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtCore import QThread,QSize
from PyQt5.QtGui import *
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from mpl_toolkits.mplot3d import Axes3D

base_path = ''
CSfunction = False
file_readed = False
label_setting = False
file_loaded = False
labels = []
imagedata = []
labels_dict = {}
autoencoder = []
datanumber=500
Whole_training = True
intermediate_layer_model=[]


class Thread_1(QThread):
    def __int__(self):
        super(Thread_1, self).__init__()
    def run(self):
        global layers_of_folders, file_readed, folder_list, labels, label_setting, imagedata, labels_dict, datanumber
        print(datanumber)
        labels_dict = {}
        if file_readed and label_setting:
            imagedata = []
            conc = 0
            for entry1 in folder_list[layers_of_folders - 1]:
                blob = []
                cellname = os.path.basename(os.path.dirname(entry1))  # extract cell name
                # print(cellname)
                concnames = os.path.basename(entry1)  # extract concentration
                # print(concnames)
                if concnames in labels:
                    labels_dict[conc] = concnames
                    fnamelist = glob.glob(os.path.join(entry1, '*.tif'))
                    for filename in fnamelist[0:datanumber]:
                        im = Image.open(filename)
                        imarray = np.array(im)
                        imarray = imarray[0:190, 0:190]
                        blob.append(imarray)
                    ind = np.reshape(np.arange(1, len(blob) + 1), (-1, 1))
                    blob_nparray = np.reshape(np.asarray(blob), (len(blob), blob[1].size))
                    blob_nparray = np.hstack((blob_nparray, ind, conc * np.ones((len(blob), 1))))
                    imagedata.append(np.asarray(blob_nparray, dtype=np.float32))
                    conc += 1
                    print('{}_{} is loaded, size = {}'.format(cellname, concnames, blob_nparray.shape))
            print('File Loaded,there are {} labeled files'.format(len(imagedata)))
        pass

class Training_Thread(QThread):

    def __int__(self):
        super(Training_Thread, self).__init__()

    def ACmodel(self,labels):
        input_img = Input(shape=(190, 190, 1),
                          name='input_layer')  # adapt this if using `channels_first` image data format
        x = Conv2D(128, (3, 3), padding='same', name='block1_conv2')(input_img)
        x = BatchNormalization(name='block1_BN')(x)
        x = Activation('relu', name='block1_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization(name='block2_BN')(x)
        x = Activation('relu', name='block2_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block2_pool')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization(name='block3_BN')(x)
        x = Activation('relu', name='block3_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block3_pool')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization(name='block4_BN')(x)
        x = Activation('relu', name='block4_act')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same', name='block4_pool')(x)
        x = Dropout(0.25)(x)

        cx = GlobalAveragePooling2D(name='globalMax')(x)
        cx = Dropout(0.5)(cx)
        class_output = Dense(len(labels), activation='softmax', name='class_output')(cx)

        x = UpSampling2D((2, 2), name='block7_upsample')(x)
        x = Conv2D(32, (3, 3), padding='same', name='block7_conv2')(x)
        x = BatchNormalization(name='block7_BN')(x)
        x = Activation('relu', name='block7_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block8_upsample')(x)
        x = Conv2D(32, (3, 3), padding='same', name='block8_conv2')(x)
        x = BatchNormalization(name='block8_BN')(x)
        x = Activation('relu', name='block8_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block9_upsample')(x)
        x = Conv2D(16, (3, 3), padding='same', name='block9_conv2')(x)
        x = BatchNormalization(name='block9_BN')(x)
        x = Activation('relu', name='block9_act')(x)
        x = Dropout(0.25)(x)

        x = UpSampling2D((2, 2), name='block10_upsample')(x)
        x = Conv2D(16, (3, 3), padding='same', name='block10_conv2')(x)
        x = BatchNormalization(name='block10_BN')(x)
        x = Activation('relu', name='block10_act')(x)
        x = Dropout(0.25)(x)
        decoded = Conv2D(1, (3, 3), name='decoder_output')(x)
        autoencoder = Model(inputs=input_img, outputs=[class_output, decoded])
        return autoencoder

    def run(self):
        K.clear_session()
        global imagedata, test_label, train_label, train_data, test_data, autoencoder, labels, CSfunction, datanumber, \
            imagesavepath, currentdate, layers_of_folders, file_readed, folder_list, label_setting, labels_dict, cavans, Whole_training, intermediate_layer_model,test_data

        for j in range(len(labels)):
            trytry = imagedata[j][:datanumber]

            # Prepare data
            LengthT = trytry.shape[0]

            for i in range(2):
                trytry = np.concatenate((trytry, trytry), axis=0)
            trytry_index = trytry[..., -2:-1]
            trytry_label = np.ones((LengthT * 4, 1)) * j  # ['Nega' for x in range(lengthN*4)] #Nega_data[...,-1:]
            trytry = trytry[..., :-2]

            # Normalize image by subtracting mean image
            trytry -= np.reshape(np.mean(trytry, axis=1), (-1, 1))
            # Reshape images
            trytry = np.reshape(trytry, (trytry.shape[0], 190, 190))

            # Rotate images
            for i in range(3):
                trytry[LengthT * (i + 1):LengthT * (i + 2)] = np.rot90(trytry[:LengthT], i + 1, (1, 2))
            # Add channel dimension to fit in Conv2D
            trytry = trytry.reshape(-1, 190, 190, 1)
            np.random.shuffle(trytry)

            trytry_train_upto = round(trytry.shape[0] * 8 / 10)
            trytry_test_upto = trytry.shape[0]

            if j is 0:
                train_data = trytry[:trytry_train_upto]
                test_data = trytry[trytry_train_upto:trytry_test_upto]
                train_label = trytry_label[:trytry_train_upto]
                test_label = trytry_label[trytry_train_upto:trytry_test_upto]

            else:
                train_data = np.concatenate((train_data,
                                             trytry[:trytry_train_upto]), axis=0)

                test_data = np.concatenate((test_data,
                                            trytry[trytry_train_upto:trytry_test_upto]), axis=0)

                train_label = np.concatenate((train_label,
                                              trytry_label[:trytry_train_upto]), axis=0)

                test_label = np.concatenate((test_label,
                                             trytry_label[trytry_train_upto:trytry_test_upto]), axis=0)

        test_label = keras.utils.to_categorical(test_label, num_classes=len(labels))
        train_label = keras.utils.to_categorical(train_label, num_classes=len(labels))
        print('Train data an test data are all prepared')

        autoencoder = self.ACmodel(labels)
        if CSfunction:
            cs_weight = 1
        else:
            cs_weight = 0
        autoencoder.compile(loss={'class_output': 'categorical_crossentropy',
                                     'decoder_output': 'mean_squared_error'},
                               loss_weights={'class_output': cs_weight, 'decoder_output': 1},
                               optimizer='Adam', metrics=['accuracy'])
        global batch_size
        batch_size = 20

        EStop = EarlyStopping(monitor='val_class_output_loss', min_delta=0,
                              patience=12, verbose=1, mode='auto')

        ReduceLR = ReduceLROnPlateau(monitor='val_class_output_loss', factor=0.1,
                                     verbose=1, patience=3, mode='auto', min_lr=1e-8)

        # Chkpnt = ModelCheckpoint('ACbin_33x128fl128GA_weights.{epoch:02d}-{val_loss:.2f}.h5',
        #                         monitor='val_class_output_loss', verbose=1, save_weights_only=True, save_best_only=True)
        global history
        history = autoencoder.fit(train_data,
                                     {'class_output': train_label, 'decoder_output': train_data},
                                     batch_size=batch_size,
                                     epochs=100, validation_split=0.25, callbacks=[EStop, ReduceLR])

        currentdate = time.strftime("%Y_%m_%d_%H_%M")
        modelsavepath = 'models/'
        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        autoencoder.save(modelsavepath + currentdate + '_Autoencoder.h5')
        global decoded_imgs
        decoded_imgs = autoencoder.predict(test_data, batch_size=batch_size, verbose=1)

        score = autoencoder.evaluate(test_data,
                                     {'class_output': test_label,
                                      'decoder_output': test_data}, batch_size=batch_size)

        # Take the bottle neck out

        intermediate_layer_model = Model(inputs=autoencoder.input,
                                         outputs=autoencoder.get_layer('globalMax').output)

        intermediate_output = intermediate_layer_model.predict(test_data, batch_size=batch_size, verbose=1)

        Y = TSNE(n_components=2, init='random', random_state=0, perplexity=30,
                 verbose=1).fit_transform(intermediate_output.reshape(intermediate_output.shape[0], -1))

        layer_output_label = np.argmax(test_label, axis=1)
        global df,Y_3D
        df = pandas.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))

        Y_3D = TSNE(n_components=3, init='random', random_state=0, perplexity=30,
                 verbose=1).fit_transform(intermediate_output)
        layer_output_label_3D = np.argmax(test_label, axis=1)
        global df_3D
        df_3D = pandas.DataFrame(dict(x=Y_3D[:, 0], y=Y_3D[:, 1], z=Y_3D[:, 2], label=layer_output_label_3D))

        global groups,groups3D
        groups = df.groupby('label')
        groups3D = df_3D.groupby('label')
        print('Start to plot t-SNE Scattering ')
        # Plot grouped scatter
        plt.cla()
        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for label, group in groups:
            name = labels_dict[label]
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name, alpha=0.8)
        plt.title('t-SNE Scattering Plot')
        ax.legend()
        imagesavepath = 'tSNE/'
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        pyplot.savefig(imagesavepath + currentdate + '_{}labels_tSNE.pdf'.format(len(labels)), format='pdf')
        pyplot.savefig(imagesavepath + currentdate + '_{}labels_tSNE.png'.format(len(labels)))

        # Save the Training History
        import collections
        historysavepath = 'history/'
        if not os.path.exists(historysavepath):
            os.makedirs(historysavepath)
        hist = history.history
        # Count the number of epoch
        for key, val in hist.items():
            numepo = len(np.asarray(val))
            break
        hist.update({'epoch': range(1, numepo + 1), 'test_class_loss': score[1],
                     'test_decoder_loss': score[2], 'test_class_acc': score[3]})
        hist = collections.OrderedDict(hist)
        hist.move_to_end('epoch', last=False)
        pandas.DataFrame(hist).to_excel(historysavepath + currentdate + '_Autoencoder_history.xlsx', index=False)
        del autoencoder
        K.clear_session()
        pass

class tSNE_Thread(QThread):

    def __int__(self):
        super(tSNE_Thread, self).__init__()

    def run(self):
        K.clear_session()

        global imagedata, test_label, train_label, train_data, test_data, autoencoder, labels, CSfunction, datanumber, \
            imagesavepath, currentdate, layers_of_folders, file_readed, folder_list, label_setting, labels_dict, cavans,Whole_training, intermediate_layer_model,test_data
        # reduce data
        # np.random.shuffle(full_data)
        for j in range(len(labels)):
            trytry = imagedata[j][:datanumber]
            # Prepare data
            LengthT = trytry.shape[0]
            # for i in range(2):
            #     trytry = np.concatenate((trytry, trytry), axis=0)
            trytry_index = trytry[..., -2:-1]
            trytry_label = np.ones((LengthT, 1)) * j  # ['Nega' for x in range(lengthN*4)] #Nega_data[...,-1:]
            trytry = trytry[..., :-2]

            # Normalize image by subtracting mean image
            trytry -= np.reshape(np.mean(trytry, axis=1), (-1, 1))
            # Reshape images
            trytry = np.reshape(trytry, (trytry.shape[0], 190, 190))

            # Rotate images
            # for i in range(3):
            #     trytry[LengthT * (i + 1):LengthT * (i + 2)] = np.rot90(trytry[:LengthT], i + 1, (1, 2))
            # Add channel dimension to fit in Conv2D
            trytry = trytry.reshape(-1, 190, 190, 1)
            np.random.shuffle(trytry)
            trytry_train_upto = round(trytry.shape[0] * 5 / 10)
            trytry_test_upto = trytry.shape[0]

            if j is 0:
                train_data = trytry[:trytry_train_upto]
                test_data = trytry[trytry_train_upto:trytry_test_upto]
                train_label = trytry_label[:trytry_train_upto]
                test_label = trytry_label[trytry_train_upto:trytry_test_upto]

            else:
                train_data = np.concatenate((train_data,
                                             trytry[:trytry_train_upto]), axis=0)

                test_data = np.concatenate((test_data,
                                            trytry[trytry_train_upto:trytry_test_upto]), axis=0)

                train_label = np.concatenate((train_label,
                                              trytry_label[:trytry_train_upto]), axis=0)

                test_label = np.concatenate((test_label,
                                             trytry_label[trytry_train_upto:trytry_test_upto]), axis=0)

        test_label = keras.utils.to_categorical(test_label, num_classes=len(labels))
        train_label = keras.utils.to_categorical(train_label, num_classes=len(labels))

        print('Train data and test data are all prepared')

        import pandas
        global model_path
        # modelsavepath = 'models/'
        # autoencoder=load_model(modelsavepath + currentdate + '_Autoencoder.h5')
        # intermediate_layer_model = Model(inputs=autoencoder.input,
        #                                  outputs=autoencoder.get_layer('globalMax').output)

        intermediate_layer_model=load_model(str(model_path[0]))
        intermediate_output = intermediate_layer_model.predict(test_data, batch_size=20, verbose=1)

        from sklearn.manifold import TSNE
        global df,df_3D,Y_3D, Y
        Y = TSNE(n_components=2, init='random', random_state=0, perplexity=30,
                 verbose=1).fit_transform(intermediate_output)
        layer_output_label = np.argmax(test_label, axis=1)
        df = pandas.DataFrame(dict(x=Y[:, 0], y=Y[:, 1], label=layer_output_label))
        Y_3D = TSNE(n_components=3, init='random', random_state=0, perplexity=30,
                 verbose=1).fit_transform(intermediate_output)
        layer_output_label_3D = np.argmax(test_label, axis=1)
        df_3D = pandas.DataFrame(dict(x=Y_3D[:, 0], y=Y_3D[:, 1], z=Y_3D[:, 2], label=layer_output_label_3D))
        global groups, groups3D
        groups = df.groupby('label')
        groups3D = df_3D.groupby('label')
        print('Start to plot t-SNE Scattering ')
        # Plot grouped scatter
        plt.cla()
        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for label, group in groups:
            name = labels_dict[label]
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name, alpha=0.8)
        plt.title('t-SNE Scattering Plot')
        ax.legend()
        currentdate2 = time.strftime("%Y_%m_%d_%H_%M")
        imagesavepath = 'tSNE/'
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        pyplot.savefig(imagesavepath + currentdate2 + '_{}labels_Re_tSNE.pdf'.format(len(labels)), format='pdf')
        pyplot.savefig(imagesavepath + currentdate2 + '_{}labels_Re_tSNE.png'.format(len(labels)))

        pass

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.actionAutoencoder.setChecked(True)
        # Connect the menu action
        self.actionSelect_Data_Path.triggered.connect(self.selectPath)
        self.actionLoad_from_Data_Path.triggered.connect(self.loadtifFile)
        # Connect the model selections
        self.actionAutoencoder.triggered.connect(self.createAE_Model)
        self.actionAE_CS.triggered.connect(self.createAECS_Model)
        self.actionOpen_Model.triggered.connect(self.openModel)
        # Connect the treeView action
        self.treeView.clicked.connect(self.on_treeView_clicked)
        # Connect the Label action
        self.actionAddLabel.triggered.connect(self.addCategory)
        self.actionDel_Label.triggered.connect(self.delCategory)
        self.actionUpdate_Setting.triggered.connect(self.updateLabelSetting)

        # Connect the Function action
        self.actionStart_Training.triggered.connect(self.Training)
        self.actionStop_Training.triggered.connect(self.stopTraining)
        self.actionNew_tSNE.triggered.connect(self.New_tSNE)
        # Gif
        self.movie = QtGui.QMovie("cube.gif")
        self.movie.setCacheMode(QtGui.QMovie.CacheAll)
        self.movie.setSpeed(80)
        self.movie.setScaledSize(QSize(256,320))
        self.Gif.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()

        import wmi

        computer = wmi.WMI()
        computer_info = computer.Win32_ComputerSystem()[0]
        os_info = computer.Win32_OperatingSystem()[0]
        proc_info = computer.Win32_Processor()[0]
        gpu_info = computer.Win32_VideoController()[0]

        os_name = os_info.Name.encode('utf-8').split(b'|')[0]
        os_version = ' '.join([os_info.Version, os_info.BuildNumber])
        system_ram = float(os_info.TotalVisibleMemorySize) / 1048576  # KB to GB

        print('OS Name: {0}'.format(os_name))
        print('OS Version: {0}'.format(os_version))
        print('CPU: {0}'.format(proc_info.Name))
        print('RAM: {0} GB'.format(system_ram))
        print('Graphics Card: {0}'.format(gpu_info.Name))

        self.setWindowIcon(QIcon('Icon.png'))
        self.System_textEdit.setText(
            'OS Name: {}\n'
            'OS Version: {}\n'
            'CPU: {}\n'
            'RAM: {} GB\n'
            'Graphics Card: {}'.format(os_name.decode("utf-8"), os_version, proc_info.Name, system_ram, gpu_info.Name))

    def Training(self):
        global file_loaded
        if file_loaded:
            self.label_7.setText('Training')
            self.label_7.setStyleSheet("color: rgb(200, 0, 0);\n"
                                       "font: 75 14pt \"MS Shell Dlg 2\";")
            self.movie.setSpeed(100)
            self.movie.start()
            self.training_thread = Training_Thread()
            self.training_thread.start()
            self.training_thread.finished.connect(self.DrawtSNE)
            self.training_thread.finished.connect(self.DrawLog)
            self.training_thread.quit()
        else:
            pass

    def New_tSNE(self):
        self.movie.start()
        self.tSNE_thread = tSNE_Thread()
        self.tSNE_thread.start()
        self.tSNE_thread.finished.connect(self.DrawtSNE)
        pass

    def stopTraining(self):
        self.training_thread = Training_Thread()
        if self.training_thread.isRunning():
            self.training_thread.terminate()
            self.movie.stop()
            self.label_7.setText('Stopped')
            self.label_7.setStyleSheet("color: rgb(200, 0, 0);\n"
                                       "font: 75 14pt \"MS Shell Dlg 2\";")

    def DrawtSNE(self):
        global imagesavepath, currentdate, labels, cavans, groups, labels_dict, groups3D,test_data,test_annotation
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        global df, df_3D, Y, Y_3D
        print("DrawtSNE Start ")
        test_annotation = test_data.reshape(-1, 190, 190)
        # 2D tSNE
        plt.cla()
        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        points_with_annotation = []
        for label, group in groups:
            name = labels_dict[label]
            point, =ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name, alpha=0.5)
            points_with_annotation.append([point])
        plt.title('t-SNE Scattering Plot')
        ax.legend()
        cavans2D = FigureCanvas(fig)
        #Annotation

        # create the annotations box
        im = OffsetImage(test_annotation[0, :, :], zoom=0.25, cmap='gray')
        xybox = (10., 10.)
        ab = AnnotationBbox(im, (10, 10), xybox=xybox, xycoords='data',
                            boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))

        # add it to the axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        tsneprelabel = int(len(test_data) / len(labels))

        def hover(event):
            global df,test_annotation
            i = 0
            ispointed = np.zeros((len(groups),),dtype=bool)
            for point in points_with_annotation:
                if point[0].contains(event)[0]:
                    ispointed[i]=True
                    cont, ind = point[0].contains(event)
                    image_index = ind["ind"][0] + i * tsneprelabel
                    # get the figure size
                    w, h = fig.get_size_inches() * fig.dpi
                    ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
                    hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
                    # if event occurs in the top or right quadrant of the figure,
                    # change the annotation box position relative to mouse.
                    ab.xybox = (xybox[0] * ws, xybox[1] * hs)
                    # place it at the position of the hovered scatter point
                    global df, test_annotation
                    df=df
                    ab.xy = (df['x'][image_index], df['y'][image_index])
                    # set the image corresponding to that point
                    im.set_data(test_annotation[image_index, :, :])
                    ab.set_visible(True)
                else:
                    ispointed[i]=False
                i = i + 1
            ab.set_visible(max(ispointed))
            fig.canvas.draw_idle()

        cid = fig.canvas.mpl_connect('motion_notify_event', hover)
        rows = int(self.tSNE_Layout.count())
        if rows == 1:
            myWidget = self.tSNE_Layout.itemAt(0).widget()
            myWidget.deleteLater()
        self.tSNE_Layout.addWidget(cavans2D)

        print("tSNE 2D Finished")
        # 3D tSNE
        fig_3D = plt.figure()
        cavans3D = FigureCanvas(fig_3D)
        ax_3D = Axes3D(fig_3D)
        ax_3D.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        test_annotation = test_data.reshape(-1, 190, 190)
        for label, group in groups3D:
            name = labels_dict[label]
            ax_3D.scatter(group.x, group.y, group.z, marker='o', label=name, alpha=0.8)
        ax_3D.legend()
        ax_3D.patch.set_visible(False)
        ax_3D.set_axis_off()
        ax_3D._axis3don = False
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        im_3D = OffsetImage(test_annotation[0, :, :], zoom=0.25, cmap='gray')
        xybox = (10., 10.)
        ab_3D = AnnotationBbox(im_3D, (10, 10), xybox=xybox, xycoords='data',
                               boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))
        # add it to the axes and make it invisible
        ax_3D.add_artist(ab_3D)
        ab_3D.set_visible(False)


        def onMouseMotion(event):
            global Y_3D
            distances = []
            for i in range(Y_3D.shape[0]):
                x2, y2, _ = proj3d.proj_transform(Y_3D[i, 0], Y_3D[i, 1], Y_3D[i, 2], ax_3D.get_proj())
                x3, y3 = ax_3D.transData.transform((x2, y2))
                distance = np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)
                distances.append(distance)
            closestIndex = np.argmin(distances)
            print(closestIndex)
            x2, y2, _ = proj3d.proj_transform(Y_3D[closestIndex, 0], Y_3D[closestIndex, 1], Y_3D[closestIndex, 2],ax_3D.get_proj())
            ab_3D.xy = (x2, y2)
            im_3D.set_data(test_annotation[closestIndex, :, :])
            ab_3D.set_visible(True)
            fig_3D.canvas.draw_idle()

        cid3d=fig_3D.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion


        rows = int(self.tSNE3D_Layout.count())
        if rows == 1:
            myWidget = self.tSNE3D_Layout.itemAt(0).widget()
            myWidget.deleteLater()
        self.tSNE3D_Layout.addWidget(cavans3D)
        self.movie.stop()
        self.movie.jumpToFrame(0)
        self.label_7.setText('Finished')
        self.label_7.setStyleSheet("color: rgb(70, 70, 70);\n"
                                   "font: 75 14pt \"MS Shell Dlg 2\";")

    def DrawLog(self):
        global history,CSfunction
        if CSfunction:
            train_history = history
            loss = 'class_output_loss'
            fig = plt.figure()
            plt.plot(range(1, len(train_history.history[loss]) + 1), train_history.history[loss])
            plt.title('Train History')
            plt.ylabel(loss)
            plt.xlabel('Epoch')
            plt.legend([loss])
            loss_cavans = FigureCanvas(fig)
            rows = int(self.Log_Layout.count())
            if rows == 1:
                myWidget = self.Log_Layout.itemAt(0).widget()
                myWidget.deleteLater()
            self.Log_Layout.addWidget(loss_cavans)
        else:
            train_history = history
            loss = 'decoder_output_loss'
            fig = plt.figure()
            plt.plot(range(1, len(train_history.history[loss]) + 1), train_history.history[loss])
            plt.title('Train History')
            plt.ylabel(loss)
            plt.xlabel('Epoch')
            plt.legend([loss])
            loss_cavans = FigureCanvas(fig)
            rows = int(self.Log_Layout.count())
            if rows == 1:
                myWidget = self.Log_Layout.itemAt(0).widget()
                myWidget.deleteLater()
            self.Log_Layout.addWidget(loss_cavans)

    def DrawDecoded(self):
        global test_data,decoded_imgs
        n = 5
        fig = plt.figure()
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(test_data[ i].reshape(190, 190))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[1][i].reshape(190, 190))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.subplot(2, n, round(n / 2))
        plt.tight_layout()
        # plt.title('Raw Images &Decoded Images')
        cavans = FigureCanvas(fig)

        # rows = int(self.Decoded_layout.count())
        # if rows == 1:
        #     myWidget = self.Decoded_layout.itemAt(0).widget()
        #     myWidget.deleteLater()
        # self.Decoded_layout.addWidget(cavans)

    def addCategory(self):
        global folder_layers, layers_of_folders,file_readed, label_setting, file_loaded
        if file_readed:
            cat_cont = int(self.Category_Layout.count()/2)
            location=self.Category_Layout.count()
            self.Category_new = QtWidgets.QLabel(self.formLayoutWidget)
            self.Category_new.setText('Label {} :'.format(cat_cont+1))
            self.Category_new.setStyleSheet("font: 14pt \"Lucida Console\";")
            self.Category_new.setObjectName("Category_{}".format(cat_cont+1))
            self.Category_Layout.setWidget(location, QtWidgets.QFormLayout.LabelRole, self.Category_new)
            self.Category_box_new = QtWidgets.QComboBox(self.formLayoutWidget)
            self.Category_box_new.setStyleSheet("background-color: rgb(255, 255, 255);\n" "color: rgb(65, 65, 65);")
            self.Category_box_new.setObjectName("Category_box{}".format(cat_cont+1))
            self.Category_Layout.setWidget(location, QtWidgets.QFormLayout.FieldRole, self.Category_box_new)
            self.Category_box_new.clear()
            for a in range(0,layers_of_folders):
                self.Category_box_new.addItems(folder_layers[a])
            self.label_3.setStyleSheet(
                "color: rgb(200, 200, 200);\n""font: 75 14pt \"MS Shell Dlg 2\";")  # Light the Label
            label_setting = False
            file_loaded = False

    def delCategory(self):
        global label_setting,file_loaded
        rows=int(self.Category_Layout.count()/1)
        if rows/2 >2:
            box = self.Category_Layout.itemAt(rows-2,QtWidgets.QFormLayout.FieldRole)
            label = self.Category_Layout.itemAt(rows-2, QtWidgets.QFormLayout.LabelRole)
            box.widget().deleteLater()
            label.widget().deleteLater()
            self.label_3.setStyleSheet(
                "color: rgb(200, 200, 200);\n""font: 75 14pt \"MS Shell Dlg 2\";")  # Light the Label
            label_setting = False
            file_loaded = False

    def updateLabelSetting(self):
        global labels, label_setting
        if str(self.Category_box1.currentText()):
            label_numbers = int(self.Category_Layout.count()/2)
            print('\nThere are {} labels.'.format(label_numbers))
            rows = int(self.Category_Layout.count())
            labels=[]
            i = 0
            for index in range(0, rows, 2):
                box = self.Category_Layout.itemAt(index, QtWidgets.QFormLayout.FieldRole)
                print('Label {} :{}'.format(i+1, str(box.widget().currentText())))
                labels.append(str(box.widget().currentText()))
                i = i+1

            self.label_3.setText('Label Updated')
            self.label_3.setStyleSheet(
                "color: rgb(70, 70, 70);\n""font: 75 14pt \"MS Shell Dlg 2\";")  # Light the Label
            label_setting = True
            print(labels)

    def readimgFile(self):
        global base_path, folder_layers, layers_of_folders, file_readed, folder_list
        if base_path:
            folder_layers = []
            folder_list = []
            layers_of_folders = 0
            files = os.scandir(base_path)
            # %% Get the 1st layer of folder
            first_folder = []
            first_folder_kind = []
            for entry in files:
                if entry.is_dir():
                    first_folder.append(entry.path)
                    first_folder_kind.append(entry.name)
            folder_layers.append(first_folder_kind)
            folder_list.append(first_folder)
            # %% Get the 2nd layer of folder
            second_folder = []
            if first_folder:
                second_folder = []
                second_folder_kind = []
                layers_of_folders += 1
                for fldr in first_folder:
                    files = os.scandir(fldr)
                    for entry in files:
                        if entry.is_dir():

                            second_folder.append(entry.path)
                            second_folder_kind.append(entry.name)
                second_folder_kind = second_folder_kind[0:int(len(second_folder_kind) / len(first_folder_kind))]
                folder_layers.append(second_folder_kind)
                folder_list.append(second_folder)
            # %% Get the 3rd layer of folder
            third_folder = []
            if second_folder:
                third_folder = []
                third_folder_kind = []
                layers_of_folders += 1
                for fldr in second_folder:
                    files = os.scandir(fldr)
                    for entry in files:
                        if entry.is_dir():
                            third_folder.append(entry.path)
                            third_folder_kind.append(entry.name)
                third_folder_kind = third_folder_kind[
                                    0:int(len(third_folder_kind) / (len(second_folder_kind) * len(first_folder_kind)))]
                folder_layers.append(third_folder_kind)
                folder_list.append(third_folder)
            # %% Get the 4th layer of folder
            forth_folder = []
            if third_folder:
                forth_folder = []
                forth_folder_kind = []
                layers_of_folders += 1
                for fldr in third_folder:
                    files = os.scandir(fldr)
                    for entry in files:
                        if entry.is_dir():
                            forth_folder.append(entry.path)
                            forth_folder_kind.append(entry.name)
                forth_folder_kind = forth_folder_kind[0:int(len(forth_folder_kind) / (
                            len(third_folder_kind) * len(second_folder_kind) * len(first_folder_kind)))]
                folder_layers.append(forth_folder_kind)
                folder_list.append(forth_folder)
            # %% Get the 5th layer of folder
            fifth_folder = []
            if forth_folder:
                fifth_folder = []
                fifth_folder_kind = []
                layers_of_folders += 1
                for fldr in third_folder:
                    files = os.scandir(fldr)
                    for entry in files:
                        if entry.is_dir():
                            fifth_folder.append(entry.path)
                            fifth_folder_kind.append(entry.name)
                fifth_folder_kind = fifth_folder_kind[0:int(len(fifth_folder_kind) / (
                            len(forth_folder_kind) * len(third_folder_kind) * len(second_folder_kind) * len(
                        first_folder_kind)))]
                folder_layers.append(fifth_folder_kind)
                folder_list.append(fifth_folder)
            print('\nThere are {} Layers in the Data Path.'.format(layers_of_folders))


            import fnmatch
            file_list = []
            for root, dirs, files in os.walk(base_path, topdown=False):
                for name in files:
                    if fnmatch.fnmatch(name, '*.tif'):
                        file_list.append(os.path.join(root, name))
            print('\nThere are {} .Tif files in the Data Path.'.format(len(file_list)))

            rows = int(self.Category_Layout.count())
            print(rows)
            for index in range(0,rows,2):
                print(index)
                box = self.Category_Layout.itemAt(index, QtWidgets.QFormLayout.FieldRole)
                print(box.widget().objectName())
                box.widget().clear()
                for a in range(0,layers_of_folders):
                    for labels in folder_layers[a]:
                        box.widget().addItem(labels)
            file_readed=True

    def loadtifFile(self):

        global datanumber,file_loaded
        datanumber = int(self.spinBox.value())
        self.thread=Thread_1()
        self.label_5.setText('Data Loading')
        self.label_5.setStyleSheet("color: rgb(200, 0, 0);\n""font: 75 14pt \"MS Shell Dlg 2\";")  # Light the Label
        self.thread.start()
        self.thread.finished.connect(self.DataCreated)
        self.thread.quit()
        file_loaded = True

    def DataCreated(self):
        self.label_5.setText('Data Loaded')
        self.label_5.setStyleSheet("color: rgb(70, 70, 70);\n""font: 75 14pt \"MS Shell Dlg 2\";")  # Light the Label

    def selectPath(self):

        global base_path
        repfldr = []
        celltypefldr = []
        self.statusBar().showMessage('Now Loading Files')
        base_path = str(QFileDialog.getExistingDirectory(self, "Select Directory", '/'))
        print(base_path)
        if base_path :
            files = os.scandir(base_path)

            # %% Get the 1st layer of folder
            for entry in files:
                if entry.is_dir():
                    if 'replicate' in entry.name:
                        repfldr.append(entry.path)
            # %% Get the 2nd layer of folder
            for fldr in repfldr:
                files = os.scandir(fldr)
                for entry in files:
                    if entry.is_dir():
                        celltypefldr.append(entry.path)
            self.readimgFile()
        self.model = QFileSystemModel()
        self.model.setRootPath(base_path)
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(base_path))
        self.treeView.hideColumn(2)
        self.treeView.hideColumn(3)

        return base_path

    def keyPressEvent(self, e):   # Use ESC to Close the APP
        from PyQt5.QtCore import Qt

        if e.key() == Qt.Key_Escape:
            self.close()

    def on_treeView_clicked(self, index):

        indexItem = self.model.index(index.row(), 0, index.parent())
        fileName = self.model.fileName(indexItem)
        filePath = self.model.filePath(indexItem)
        from PyQt5.QtGui import QIcon, QPixmap
        pixmap = QPixmap(filePath)
        self.Cell_Image.setPixmap(pixmap.scaled(256,256))

    def createAE_Model(self, state):
        global CSfunction
        if state:
            self.actionAE_CS.setChecked(False)
            CSfunction = False

    def createAECS_Model(self, state):
        global CSfunction
        if state:
            self.actionAutoencoder.setChecked(False)
            CSfunction = True

    def openModel(self):
        global model_path
        model_path= QFileDialog.getOpenFileName(self, "Select Model", '/')
        print(model_path)




    def runExample(self):
        global base_path, folder_layers, layers_of_folders, file_readed, folder_list




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

