import sys, os, io
home_dir = os.getcwd()
sys.path.append(home_dir)
import tensorflow as tf


def restore_and_save(input_checkpoint, export_path_base):
    checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print (graph.get_name_scope())

            export_path_base = export_path_base
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(count)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和我们需要的结果（project/logits,crf_loss/transitions）
            """
            build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
            输入：tensorflow graph中的tensor；
            输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
                        get_operation_by_name：通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
            """

            char_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("ChatInputs").outputs[0])
            seg_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("SegInputs").outputs[0])
            dropout = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("Dropout").outputs[0])
            logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("project/logits").outputs[0])

            transition_params = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("crf_loss/transitions").outputs[0])

            """
            signature_constants：SavedModel保存和恢复操作的签名常量。
            在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
            """
            # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
            labeling_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "charinputs":
                            char_inputs,
                        "dropout":
                            dropout,
                        "seginputs":
                            seg_inputs,
                    },
                    outputs={
                        "logits":
                            logits,
                        "transitions":
                            transition_params
                    },
                    method_name="tensorflow/serving/predict"))

            """
            tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                          输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                          对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                          对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                       labeling_signature},
                legacy_init_op=legacy_init_op)

            builder.save()
            print("Build Done")

### 测试模型转换
tf.flags.DEFINE_string("ckpt_path",     "source_ckpt/IDCNN",             "path of source checkpoints")
tf.flags.DEFINE_string("pb_path",       "servable-models/IDCNN",             "path of servable models")
tf.flags.DEFINE_integer("version",      1,              "the number of model version")
tf.flags.DEFINE_string("classes",       'LOC',          "multi-models to be converted")
FLAGS = tf.flags.FLAGS

classes = FLAGS.classes
input_checkpoint = FLAGS.ckpt_path + "/" + classes
model_path = FLAGS.pb_path + '/' + classes

# 版本号控制
count = FLAGS.version
modify = False
if not os.path.exists(model_path):
    os.mkdir(model_path)
else:
    for v in os.listdir(model_path):
        print(type(v), v)
        if int(v) >= count:
            count = int(v)
            modify = True
    if modify:
        count += 1

# 模型格式转换
restore_and_save(input_checkpoint, model_path)



channel = implementations.insecure_channel("127.0.0.1", 8500)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()

# 指定启动tensorflow serving时配置的model_name和是保存模型时的方法名
request.model_spec.name = "model1"
request.model_spec.signature_name = "serving_default"