import pickle

def load_model(filename='best_model_try.pkl'):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# 加载模型
model_data_stab = load_model('best_model_try.pkl')
model_data_swing = load_model('best_model.pkl')
# 从加载的模型数据中获取controller
controller_stab = model_data_stab['controller']
controller_swing = model_data_swing['controller']
# 打印controller的内容
# 注意：这个步骤取决于controller对象的类型和结构
print("Controller_stab data:")
print(controller_stab)
print("Controller_swing data:")
print(controller_swing)
# 如果controller是一个复杂的对象，你可能需要遍历它的属性或使用特定的方法来查看更详细的信息
# 例如，如果controller有一个可以打印其内容的方法：
print(dir(controller_stab))
print(dir(controller_swing))
if hasattr(controller_stab, 'X'):
    print("X_stab:", controller_stab.X)

if hasattr(controller_swing, 'X'):
    print("X_swing:", controller_swing.X)

if hasattr(controller_stab, 'Y'):
    print("Y_stab:", controller_stab.Y)

if hasattr(controller_swing, 'Y'):
    print("Y_swing:", controller_swing.Y)

if hasattr(controller_stab, 'lengthscales'):
    print("lengthscales_stab:", controller_stab.lengthscales)

if hasattr(controller_swing, 'lengthscales'):
    print("lengthscales_swing:", controller_swing.lengthscales)
#
# if hasattr(controller, 'noise'):
#     print("Noise:", controller.noise)
#
# if hasattr(controller, 'variance'):
#     print("Variance:", controller.variance)