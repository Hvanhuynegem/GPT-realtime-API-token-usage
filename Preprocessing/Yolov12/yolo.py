from ultralytics import YOLO
import onnx

model = YOLO("yolo12n.pt")
model.export(format="onnx", opset=21)

m = onnx.load("yolo12n.onnx")
print("op version: (... , Version)", [(op.domain, op.version) for op in m.opset_import])