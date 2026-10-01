import os
import argparse
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models.solar_cnn import SolarCNN

def main():
    parser = argparse.ArgumentParser(description="Export SolarCast model to ONNX")
    parser.add_argument('--model-path', type=str, default='models/best_model.pth')
    parser.add_argument('--output-path', type=str, default='models/solar_cnn.onnx')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    model = SolarCNN()
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"Weights not found at {args.model_path}, exporting untrained model.")
        
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model successfully exported to {args.output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed.")
    except ImportError:
        print("Install 'onnx' to verify the exported model.")

if __name__ == '__main__':
    main()
