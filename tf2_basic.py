import numpy as np
import seaborn as sns
import tensorflow as tf

iris = sns.load_dataset('iris')

print(iris.head())

# แบ่งข้อมูลเป็นเทรนและเทสโดยใช้ 25 ตัวแรกของแต่ละชนิดเป็นข้อมูลทดสอบ
train_set = iris.groupby('species').head(25)

# หาชุดข้อมูลสำหรับการฝึกโดยใช้ตัวอย่างที่ไม่ใช่ข้อมูลทดสอบ
test_set = iris.drop(train_set.index)

# เตรียมข้อมูลเซ็ตเทรนและเซ็ตทดสอบ
X_train = train_set.drop('species', axis=1)
y_train = train_set['species']

X_test = test_set.drop('species', axis=1)
y_test = test_set['species']

# รหัสของคลาส
class_codes = {
    'setosa': [1, 0, 0],
    'versicolor': [0, 1, 0],
    'virginica': [0, 0, 1]
}


def encode_classes(y, class_codes):
    return np.array([class_codes[label] for label in y])


# แปลงรหัสคลาสในชุดข้อมูลการฝึก
y_train_encoded = encode_classes(y_train, class_codes)

# แปลงรหัสคลาสในชุดข้อมูลทดสอบ
y_test_encoded = encode_classes(y_test, class_codes)

# ตรวจสอบว่าการแปลงเป็น One-Hot เรียบร้อย
print("Encoded Training Labels:\n", y_train_encoded[:5])
print("Encoded Testing Labels:\n", y_test_encoded[:5])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, input_shape=(4,)))
model.add(tf.keras.layers.Activation('sigmoid'))
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Activation('sigmoid'))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(X_train, y_train_encoded, epochs=1000)

print('Model Trained!')

# ทำการทำนายผลลัพธ์จากชุดข้อมูลทดสอบ
y_pred = model.predict(X_test)

# แปลงผลลัพธ์ที่ทำนายให้เป็นรหัสคลาส
y_pred_decoded = np.argmax(y_pred, axis=1)
y_test_decoded = np.argmax(y_test_encoded, axis=1)

# หาค่าความแม่นยำ
accuracy = np.mean(y_pred_decoded == y_test_decoded)
print("Accuracy:", accuracy)
