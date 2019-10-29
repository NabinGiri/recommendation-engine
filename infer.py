import sys
import os
import numpy as np
import pickle
import math

RANDOM_ORDER_FORMAT = 0
USER_GROUP_FORMAT = 1
AUTO_DETECT = 2

LINEAR = 'LINEAR'
SIGMOID_L2 = "SIGMOID_L2"
SIGMOID_LIKELIHOOD = "SIGMOID_LIKELIHOOD"
SIGMOID_RANK = "SIGMOID_RANK"
HINGE_SMOOTH = "HINGE_SMOOTH"
HINGE_L2 = "HINGE_L2"
SIGMOID_QSGRAD = "SIGMOID_QSGRAD"


def map_active(total, activation):
    if activation == LINEAR:
        return total
    elif activation == SIGMOID_L2:
        return None
    elif activation == SIGMOID_LIKELIHOOD:
        return 1.0 / (1.0 + math.exp(-total))
    elif activation == SIGMOID_RANK:
        return total
    elif activation == HINGE_SMOOTH:
        return total
    elif activation == HINGE_L2:
        return total
    elif activation == SIGMOID_QSGRAD:
        return total
    else:
        print("Unknown Active Type")
        return 0.0


def active_type(name, value):
    active = None
    if name == "active_type":
        if value == '0':
            active = LINEAR
        elif value == '1':
            active = SIGMOID_L2
        elif value == '2':
            active = SIGMOID_LIKELIHOOD
        elif value == '3':
            active = SIGMOID_RANK
        elif value == '5':
            active = HINGE_SMOOTH
        elif value == '6':
            actve = HINGE_L2
        elif value == '7':
            active = SIGMOID_QSGRAD
    return active


class config:
    def __init__(self):
        self.cfg = []
        self.tmp2 = []
        self.tmp1 = []
        self.name_config = "config.conf"
        with open(self.name_config) as file:
            for line in file:
                line_value = line.split()
                self.tmp2.append(line_value)

        for i in range(0, len(self.tmp2)):
            if len(self.tmp2[i]) < 4 and len(self.tmp2[i]) != 0:
                self.tmp1.append(self.tmp2[i])

        for i in range(0, len(self.tmp1)):
            name = self.tmp1[i][0]
            value = self.tmp1[i][2]
            self.cfg.append((name, value))
        file.close()


class SVDTypeParam:
    def __init__(self):
        # try to decide the format.
        self.format_type = AUTO_DETECT
        self.active_type = self.extend_type = self.variant_type = 0
        self.config = config()
        for name, value in self.config.cfg:
            self.set_param_type(name, value)
        self.decide_format()

    def set_param_type(self, name, value):
        if name == "model_type":
            self.model_type = value
        if name == "format_type":
            self.format_type = value
        if name == "active_type":
            self.active_type = value
        if name == "extend_type":
            self.extend_type = value
        if name == "variant_type":
            self.variant_type = value

    def decide_format(self):
        if self.format_type != AUTO_DETECT:
            return None
        else:
            self.format_type = RANDOM_ORDER_FORMAT if self.extend_type == '1' else USER_GROUP_FORMAT
            return self.format_type


def create_svd_trainer():
    mtype = SVDTypeParam()
    if mtype.extend_type == "15":
        return SVDBiLinearTrainer()
    if mtype.extend_type == "2":
        return SVDPPMultiIMFB()
    if mtype.extend_type == "30":
        return APLambdaGBRTTrainer()
    if mtype.extend_type == "31":
        return RegGBRTTrainer()
    if mtype.extend_type == "1":
        return SVDFeature()
    if mtype.format_type == USER_GROUP_FORMAT or RANDOM_ORDER_FORMAT:
        return SVDPPFeature()


class SVDBiLinearTrainer:
    pass


class SVDPPMultiIMFB:
    pass


class APLambdaGBRTTrainer:
    pass


class RegGBRTTrainer:
    pass


class SVDPPFeature:
    pass


class SVDModelParam:

    def __init__(self):
        self.num_user = self.num_item = self.num_global = self.num_factor = 0
        self.u_init_sigma = self.i_init_sigma = float(0.01)  # std variance for user and item factor
        self.no_user_bias = float(0.0)
        # global mean of prediction
        self.base_score = float(0.5)
        self.num_ufeedback = 0
        self.ufeedback_init_sigma = float(0.0)
        self.num_randinit_ufactor = self.num_randinit_ifactor = 0
        self.common_latent_space = 0
        self.user_nonnegative = 0
        self.item_nonnegative = 0
        self.common_feedback_space = 0
        self.extend_flag = 0
        self.config = config()
        self._init_end = 0
        for name, value in self.config.cfg:
            self.set_param(name, value)

    def set_param(self, name, value):
        if name == "num_user":
            self.num_user = value
        elif name == "num_item":
            self.num_item = value
        elif name == "num_uiset":
            self.num_user = self.num_item = value
        elif name == "num_global":
            self.num_global = value
        elif name == "num_factor":
            self.num_factor = value
        elif name == "u_init_sigma":
            self.u_init_sigma = value
        elif name == "i_init_sigma":
            self.i_init_sigma = value
        elif name == "ui_init_sigma":
            self.u_init_sigma = self.i_init_sigma = value
        elif name == "base_score":
            self.base_score = value
        elif name == "no_user_bias":
            self.no_user_bias = value
        elif name == "num_ufeedback":
            self.num_ufeedback = value
        elif name == "num_randinit_ufactor":
            self.num_randinit_ufactor = value
        elif name == "num_randinit_ifactor":
            self.num_randinit_ifactor = value
        elif name == "num_randinit_uifactor":
            self.num_randinit_ufactor = self.num_randinit_ifactor = value
        elif name == "ufeedback_init_sigma":
            self.ufeedback_init_sigma = value
        elif name == "common_latent_space":
            self.common_latent_space = value
        elif name == "common_feedback_space":
            self.common_feedback_space = value
        elif name == "user_nonnegative":
            self.user_nonnegative = value
        elif name == "item_nonnegative":
            self.item_nonnegative = value


class SVDFeature:

    def __init__(self):
        self.__name_feat_user = None
        self.__name_feat_item = None
        self._feat_user = []
        self._feat_item = []
        self._init_end = 0
        self.config = config()
        self.init_trainer()
        for name, value in self.config.cfg:
            if name == "active_type":
                active_type(name, value)
                break
        self.active_type = active_type(name, value)

    def init_trainer(self):
        if self.__name_feat_user is not None:
            self._feat_user.append(self.__name_feat_user)
        if self.__name_feat_item is not None:
            self._feat_item.append(self.__name_feat_item)
        self._sample_counter = 0
        # if self._param.reg_global >= 4:
        # self.ref_global = []
        # if self._param.reg_method >= 4:
        # self.ref_user = []
        # if self._model.param.common_latent_space == 0:
        # self.ref_item = []
        # else:
        # self.ref_item = self.ref_user
        self._init_end = 1

    def load_from_file(self, filename):
        if type(filename) == str:
            model_file = filename
        else:
            model_file = filename.name
        if os.stat(model_file).st_size == 0:
            print("Error loading CF SVD model")
            exit(1)
        try:
            file = open(model_file, "rb")
        except IOError:
            print("There is no model file")
        # self.mtype = pickle.load(file)
        # self.param = pickle.load(file)
        if SVDModelParam().common_latent_space == 0:
            self.u_bias = pickle.load(file)
            assert len(self.u_bias) > 0, "load from file"
            self.W_user = pickle.load(file)
            self.i_bias = pickle.load(file)
            self.W_item = pickle.load(file)
        else:
            self.ui_bias = pickle.load(file)
            self.W_uiset = pickle.load(file)
        self.g_bias = pickle.load(file)
        if SVDTypeParam().format_type == USER_GROUP_FORMAT:
            if SVDModelParam().common_feedback_space:
                self.ufeedback_bias = pickle.load(file)
                self.W_ufeedback = pickle.load(file)
        self.tmp_ufactor = self.W_user[0]
        self.tmp_ifactor = self.W_item[0]

    def predict(self, i):
        total = float(int(SVDModelParam().base_score)) + (
            self.calc_bias(i, self.u_bias, self.i_bias, self.g_bias))
        self.prepare_tmp(i)
        total += np.dot(self.tmp_ufactor, self.tmp_ifactor)
        return map_active(total, self.active_type)

    def prepare_tmp(self, i):
        # self.svdpp._prepare_svdpp()
        # for i in range(len(self.feature.elems)):
        for j in range(i.num_ufactor[0]):
            uid = int(i.index_ufactor[j])
            assert uid < int(SVDModelParam().num_user), "user feature index exceed bound"
            value_ufactor = int(i.value_ufactor[j])
            value = np.multiply(self.W_user[uid], value_ufactor)
            self.tmp_ufactor += value

        # for i in range(len(self.feature.elems)):
        for j in range(i.num_ifactor[0]):
            iid = int(i.index_ifactor[j])
            ival = int(i.value_ifactor[j])
            val = np.multiply(self.W_item[iid], ival)
            self.tmp_ifactor += val

    def calc_bias(self, i, u_bias, i_bias, g_bias):
        total = float(0.0)
        # for i in range(len(self.feature.elems)):
        for j in range((i.num_global[0])):
            gid = i.index_global[j]
            assert gid < SVDModelParam().num_global, "global feature index exceed setting"
            total += i.value_global[j] * g_bias[gid]

        if SVDModelParam().no_user_bias == 0:
            # for i in range(len(self.feature.elems)):
            for j in range(i.num_ufactor[0]):
                uid = int(i.index_ufactor[j])
                assert uid < int(SVDModelParam().num_user), "user feature index exceed bound"
                u_bias_value = u_bias[uid]
                total += (int(i.value_ufactor[j]) * u_bias_value)

            # total += self._get_bias_svdpp()

            # for i in range(len(self.feature.elems)):
            for j in range(i.num_ifactor[0]):
                iid = int(i.index_ifactor[j])
                ival = int(i.value_ifactor[j])
                assert iid < int(SVDModelParam().num_item), "item feature index exceed bound"
                i_bias_value = i_bias[iid]
                total += ival * i_bias_value
        # total += self._get_bias_plugin()
        return total


class Elem:
    def __init__(self):
        # result label , rate or {0,1} for classification.
        self.label = float(0)
        # number of non-zero global feature
        self.num_global = 0
        # number of non-zero user feature
        self.num_ufactor = 0
        # number of non-zero item feature
        self.num_ifactor = 0
        # array of global feature index
        self.index_global = []
        # array of user feature index
        self.index_ufactor = []
        # array of item feature index
        self.index_ifactor = []
        # array of global feature value
        self.value_global = []
        # array of user feature value
        self.value_ufactor = []
        # array of item feature value
        self.value_ifactor = []
        # array of feedback
        self.feedback_id = []
        self.feedback_value = []


class SVDFeatureCSR:

    def load_from_file(self, filename):
        with open(filename, "r") as data_file:
            for line in data_file:
                elem = Elem()
                line_values = line.split()
                elem.label = [int(line_values[0])]
                elem.num_global = [int(line_values[1])]
                elem.num_ufactor = [int(line_values[2])]
                elem.num_ifactor = [int(line_values[3])]
                # index_global = [(line_values[4])]
                ufactor = [(line_values[4])]
                ifactor = [(line_values[5])]
                for i in ifactor:
                    value_i = i.split(":")
                    elem.index_ifactor.append(value_i[0])
                    elem.value_ifactor.append(value_i[1])

                for i in ufactor:
                    value_u = i.split(":")
                    elem.index_ufactor.append(value_u[0])
                    elem.value_ufactor.append(value_u[1])
                self.elems.append(elem)

        with open("ua.base.feedbackfeature", "r") as file:
            for line in file:
                line_values = line.split()
                n = int(line_values[1])
                elem.num_feedback = n
                for i in range(2, n):
                    feedback_val = line_values[i]
                    f = feedback_val.split(":")
                    elem.feedback_id.append(f[0])
                    elem.feedback_value.append(f[1])
            self.elems.append(elem)
        file.close()
        data_file.close()

    def __init__(self):
        self.elems = []
        self.load_from_file(filename)
        self.num_ufeedback = 0
        self.name_data = None
        self.scale_score = float(0.0)
        self.config = config()
        for name, value in self.config.cfg:
            self.set_param(name, value)

    def set_param(self, name, value):
        if name == "scale_score":
            self.scale_score = value
        if name == "data_in":
            self.name_data = value


class RMSEEvaluator:

    def __init__(self):
        self.total = 0.0
        self.original = []
        self.tmp = []
        self.pred = []

    def rmse_calculator(self):
        file = open("ua.test.feature", "r")
        for line in file:
            self.original.append(float(line[0]))
        file_pred = open("pred.txt", "r")
        for line in file_pred:
            lines = line.split("\n")
            self.tmp.append(lines)
            for i in range(len(self.tmp)):
                self.pred.append(float(self.tmp[i][0]))
        for i in range(len(self.original)):
            val_o = (self.original[i])
            val_p = (self.pred[i])
            value = val_o - val_p
            self.total += value
        return math.sqrt(1 / len(self.original) * (math.pow(self.total, 2)))


class SVDInferTask:

    def __init__(self):
        self.config = config()
        self.step = 1
        self.pred_model = -1
        self.pred_binary = 0
        self.scale_score = 1.0
        self.init_end = 0
        self.name_pred = "pred.txt"
        self.name_model_in_folder = "models"
        self.start = 0
        self.end = sys.maxsize
        self.use_ranker = 0
        self.num_item_set = 0
        for name, value in self.config.cfg:
            self.set_param_inner(name, value)
        # Todo: create the class create_svd_ranker.
        # self.svd_ranker = create_svd_ranker()

    def set_param_inner(self, name, value):
        if name == "model_out_folder":
            self.name_model_in_folder = value
        if name == "log_eval":
            self.name_eval = value
        if name == "name_pred":
            self.name_pred = value
        if name == "start":
            self.start = value
        if name == "end":
            self.end = value
        if name == "focus":
            self.end = self.start + 1
        if name == "pred":
            self.pred_model = value
            self.end = self.start + 1
        if name == "pred_binary":
            self.pred_binary = value
        if name == "step":
            self.step = value
        if name == "silent":
            self.silent = value
        if name == "job":
            self.name_job = value
        if name == "scale_score":
            self.scale_score = value
        if name == "test":
            self.input_type = value
        if name == "use_ranker":
            self.use_ranker = value
        if name == "num_item_set":
            self.num_item_set = value

    def init(self):
        self.init_model(int(self.pred_model))
        if self.svd_inferencer is not None:
            self.svd_inferencer.init_trainer()
        self.configure_iterator()
        self.init_end = 1

    def configure_inferencer(self):
        if self.svd_inferencer is not None:
            for name, value in self.config.cfg:
                self.set_param_inner(name, value)

    def init_model(self, start):
        fi = f"{self.name_model_in_folder}/{start:04}.model"
        if self.use_ranker == 0:
            self.svd_inferencer = create_svd_trainer()
            self.svd_inferencer.load_from_file(fi)
            # fi.close()
        else:
            pass
            exit(0)

    def configure_iterator(self):
        if SVDTypeParam().format_type is not USER_GROUP_FORMAT:
            self.itr_csr = SVDFeatureCSR()

    def run_task(self):
        self.init()
        self.configure_inferencer()
        if int(self.pred_model) >= 0:
            if self.svd_inferencer is not None:
                self.task_pred()

    def fopen_pred(self):
        if self.pred_binary == 0:
            return open(self.name_pred, "w")
        else:
            return open(self.name_pred, "wb")

    def task_pred(self):
        fo = self.fopen_pred()
        if self.itr_csr is not None:
            for i in range(len(self.itr_csr.elems)):
                p = float(self.svd_inferencer.predict(self.itr_csr.elems[i]))
                #round off to 6 digits
                #prediction = round(p,6)
                self.write_pred(fo, p * self.scale_score)

    def write_pred(self, fo, pred):
        fo.write(str(pred))
        fo.write("\n")


if __name__ == '__main__':
    filename = "ua.test.feature"
    SVDInferTask().run_task()
    print("Using model number", SVDInferTask().pred_model, "to predict the ratings.\nPrediction end,result stored on",
          SVDInferTask().name_pred)
    print("Root Mean Square Error is: ", RMSEEvaluator().rmse_calculator())
