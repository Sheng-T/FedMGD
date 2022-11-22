import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):

        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):

        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=str(self.display_id) + str(idx) + self.name,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=str(self.display_id)+"|"+str(self.name))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_Ds_prediction(self, epoch, counter_ratio, preds):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_pred'):
            self.plot_data_pred = {'X': [], 'Y': [], 'legend': list(preds.keys())}
        self.plot_data_pred['X'].append(epoch + counter_ratio)
        self.plot_data_pred['Y'].append([preds[k] for k in self.plot_data_pred['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data_pred['X'])] * len(self.plot_data_pred['legend']), 1),
                Y=np.array(self.plot_data_pred['Y']),
                opts={
                    'title': self.name + ' Ds prediction over time',
                    'legend': self.plot_data_pred['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'pred'},
                win=str(self.display_id+1)+"|"+str('prediction of Ds'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_label_prediction(self, epoch, counter_ratio, preds):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_label_pred'):
            self.plot_label_pred = {'X': [], 'Y': [], 'legend': list(preds.keys())}
        self.plot_label_pred['X'].append(epoch + counter_ratio)
        self.plot_label_pred['Y'].append([preds[k] for k in self.plot_label_pred['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_label_pred['X'])] * len(self.plot_label_pred['legend']), 1),
                Y=np.array(self.plot_label_pred['Y']),
                opts={
                    'title': self.name + ' predicted label over time',
                    'legend': self.plot_label_pred['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'pred'},
                win=str(self.display_id+1)+"|"+str('predicted labels'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def plot_fc_prediction(self, epoch, counter_ratio, fc_preds):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_fc_pred'):
            self.plot_data_fc_pred = {'X': [], 'Y1': [], 'Y3': [], 'Y5': [], 'Y7': [], 'Y9': [],
                                      'legend': list(fc_preds.keys())}
        self.plot_data_fc_pred['X'].append(epoch + counter_ratio)
        for i in range(1, 10, 2):
            self.plot_data_fc_pred['Y{:d}'.format(i)].append([fc_preds[k][i].item() for k in self.plot_data_fc_pred['legend']])
            try:
                self.vis.line(
                    X=np.stack([np.array(self.plot_data_fc_pred['X'])] * len(self.plot_data_fc_pred['legend']), 1),
                    Y=np.array(self.plot_data_fc_pred['Y{:d}'.format(i)]),
                    opts={
                        'title': self.name + ' fc prediction_label=' + str(i),
                        'legend': self.plot_data_fc_pred['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'pred'},
                    win=str(self.display_id+10+i)+"|"+str('prediction of fc'))
            except VisdomExceptionBase:
                self.create_visdom_connections()

    def plot_fc_weights(self, epoch, weights):
        """display the current fc layer weights

        Parameters:
            epoch (int)           -- current epoch
            weights (Dict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data_fc_weights'):
            self.plot_data_fc_weights = {'X': [], 'legend': list(weights['names'])}
        try:
            self.vis.bar(
                X=weights['values'],
                opts={
                    'stacked': False,
                    'title': self.name + ' fc abs weights_epoch ' + str(epoch),
                    'legend': self.plot_data_fc_weights['legend']
                },
                win=str(self.display_id + epoch + 2) + "|" + str('fc weights'))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
