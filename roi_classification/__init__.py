from PySide6.QtGui import QAction
import numpy as np

from pydetecdiv import plugins
from pydetecdiv.app import PyDetecDiv, pydetecdiv_project
from pydetecdiv.app.gui.Windows import MatplotViewer
from pydetecdiv.domain.Image import Image


class Plugin(plugins.Plugin):
    id_ = 'gmgm.plewniak.extensions.roiclassification'
    version = '1.0.0'
    name = 'Deep learning'
    category = 'ROI classification'
    parent = 'gmgm.plewniak.roiclassification'

    def __init__(self):
        super().__init__()

    def addActions(self, menu):
        if self.parent_plugin:
            action_launch = QAction("Show results", self.parent_plugin.menu)
            action_launch.triggered.connect(self.launch)
            self.parent_plugin.menu.addAction(action_launch)
            action_test_model = QAction("Test model", self.parent_plugin.menu)
            action_test_model.triggered.connect(self.test_model)
            self.parent_plugin.menu.addAction(action_test_model)

    def launch(self):
        print(f'{self.parent_plugin.predictions.shape}')
        self.show_sequence()
        self.show_predictions_heatmap()

    def show_sequence(self, t=0, length=2, step=1):
        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(self.parent_plugin.roi_list),
                                    columns=int(length / step + 0.5))
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Sequences')

        for i, (pred, roi) in enumerate(zip(self.parent_plugin.predictions, self.parent_plugin.roi_list)):
            plot_viewer.axes[i, 0].set_ylabel(f'{roi.name}', fontsize='xx-small')
            for j in range(int(length / step + 0.5)):
                p = pred[j]
                max_score, max_index = max((value, index) for index, value in enumerate(p))
                plot_viewer.axes[i, j].set_title(f'{t + j} ({self.parent_plugin.class_names[max_index]})',
                                                 fontsize='xx-small')
                plot_viewer.axes[i, j].set_xticks([])
                plot_viewer.axes[i, j].set_yticks([])
                # img = Image(self.parent_plugin.img_array[i][t + j])
                with pydetecdiv_project(PyDetecDiv().project_name) as project:
                    fov = project.get_linked_objects('FOV', roi)
                    imgdata = fov[0].image_resource().image_resource_data()
                    img = Image(self.parent_plugin.get_rgb_images_from_stacks(imgdata, [roi], t+j)[0])
                    img.show(plot_viewer.axes[i, j])

        plot_viewer.canvas.draw()
        PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

    def show_predictions_heatmap(self, t=0, length=100, ):
        roi_names = [roi.name for roi in self.parent_plugin.roi_list]
        heatmap_plot = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(roi_names), )
        PyDetecDiv().main_window.active_subwindow.addTab(heatmap_plot, 'Predictions')
        for i, (pred, roi) in enumerate(zip(self.parent_plugin.predictions, roi_names)):
            heatmap_plot.axes[i].set_ylabel(f'{roi}', fontsize='xx-small')
            heatmap_plot.axes[i].imshow(np.moveaxis(pred[t:t + length, ...], 0, -1))
            heatmap_plot.axes[i].set_yticks(np.arange(len(self.parent_plugin.class_names)),
                                            labels=self.parent_plugin.class_names,
                                            fontsize='xx-small')
            heatmap_plot.axes[i].set_aspect('auto')

        heatmap_plot.canvas.draw()
        PyDetecDiv().main_window.active_subwindow.setCurrentWidget(heatmap_plot)

    def test_model(self, ):
        import tifffile
        import tensorflow as tf
        import os

        module = self.parent_plugin.gui.network.currentData()
        model = module.load_model(load_weights=False)
        weights = self.parent_plugin.gui.weights.currentData()
        if weights:
            module.loadWeights(model, filename=self.parent_plugin.gui.weights.currentData())

        images = np.array(
            [
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/small/Pos0_1_221_frame_0410.tif',
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/large/Pos0_1_83_frame_0211.tif',
                '/NAS/DataGS02/Fred/div_1_first_tests/trainingdataset/images/empty/Pos0_1_47_frame_0018.tif'
            ])

        class_names = ['clog', 'dead', 'empty', 'large', 'small', 'unbud']

        plot_viewer = MatplotViewer(PyDetecDiv().main_window.active_subwindow, rows=len(images), columns=2)
        PyDetecDiv().main_window.active_subwindow.addTab(plot_viewer, 'Predictions')

        for i, fichier in enumerate(images):
            image = Image(tifffile.imread(fichier))
            sequence = tf.stack((image.as_tensor(),), axis=0)
            # sequence = image.as_tensor()
            img_array = tf.expand_dims(sequence, 0)  # Create batch axis
            # img_array = tf.expand_dims(data, 0)  # Create batch axis
            print(img_array.shape, img_array.dtype)
            img_array = tf.convert_to_tensor([tf.image.resize(i, (224, 224), method='nearest') for i in img_array])

            print(img_array.shape, img_array.dtype)
            # print(img_array)

            data, predictions = model.predict(img_array)

            score = predictions[0, 0,]

            plot_viewer.axes[i][0].set_title(os.path.basename(fichier))
            image.show(ax=plot_viewer.axes[i][0])
            image.channel_histogram(ax=plot_viewer.axes[i][1], bins=64)
            max_score, max_index = max((value, index) for index, value in enumerate(score))
            plot_viewer.axes[i][0].text(1, 5, f'{class_names[max_index]}: {max_score:.2f}',
                                        {'fontsize': 8, 'color': 'yellow'})
            plot_viewer.canvas.draw()
            PyDetecDiv().main_window.active_subwindow.setCurrentWidget(plot_viewer)

            print(fichier)
            for c, s in enumerate(score):
                print(f'{class_names[c]}: {s:.2f}', )
