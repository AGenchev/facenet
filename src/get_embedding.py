import tensorflow as tf
import facenet
import cv2
import align.detect_face
import numpy as np

gpu_memory_fraction = 1.0
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

with tf.Graph().as_default():
    face_comp_sess = tf.Session()
    with face_comp_sess.as_default():
        facenet.load_model("../20180402-114759/")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

def detect_faces(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # for i, b in enumerate(bounding_boxes):
    #     cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0),thickness=3)
    # cv2.imwrite("output.jpg", img)
    return bounding_boxes, img


def generate_cropped_face(bounding_boxes, img):
    image_size = 160
    margin = 44
    img_list = []
    img_size = np.asarray(img.shape)[0:2]
    for i, each_face in enumerate(bounding_boxes):
        det = np.squeeze(bounding_boxes[i,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    return img_list

def main(image_path):
    bounding_boxes, img = detect_faces(image_path)
    preprocess_face_list = generate_cropped_face(bounding_boxes, img)
    generated_embeddings = face_comp_sess.run(embeddings, feed_dict={images_placeholder: np.stack(preprocess_face_list_1), phase_train_placeholder:False })
    return generated_embeddings
