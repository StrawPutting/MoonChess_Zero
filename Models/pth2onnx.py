import torch
import sys
import os

#os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DQN2 import DQN

def export_to_onnx(model_path="moonchess_policy_final.pth", onnx_path="moonchess_policy.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 16
    output_dim = 9
    
    policy_net = DQN(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    
    dummy_input = torch.randn(1, input_dim).to(device)
    
    torch.onnx.export(
        policy_net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"ONNX模型已导出至: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx(model_path="Models/moonchess_policy_400000.pth")
