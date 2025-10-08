import os
import json
import vtk

def save_vtp(poly_data, output_path):
    try:
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            return
    
        writer = vtk.vtkXMLDataObjectWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(poly_data)
        writer.Write()
        print(f"VTP file saved: {output_path}")
    except Exception as e:
        print(f"Failed to save VTP file: {output_path}, Error: {e}")

def create_vtp_from_json(json_path, output_dir):
    print(f"Processing JSON file: {json_path}")
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)

        if 'ArrayOfannotation_info' not in data:
            print(f"Key 'ArrayOfannotation_info' not found in {json_path}")
            return
        
        for item in data['ArrayOfannotation_info']:
            xyvalue = item.get('xyvalue', {})
            label_val = xyvalue.get('label_val', {})
            preset_name = label_val.get('preset_name')
            preset_detail_name = label_val.get('preset_detail_name')
            objectname = item.get('objectname', '')

            if preset_name and preset_detail_name:
                base_filename = os.path.basename(json_path).replace(".json", f"_{preset_detail_name}_{objectname}.vtp")
                output_path = os.path.join(output_dir, base_filename)

                poly_data = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

                if objectname == "LabelCenterLine":
                    print(f"Createting LabelCenterLine for {base_filename}")
                    start_c_pos = xyvalue.get('start_c_pos', None)
                    end_c_pos = xyvalue.get('end_c_pos', None)
                    if start_c_pos and end_c_pos:
                        points.InsertNextPoint(start_c_pos['X'], -start_c_pos['Y'], 0)
                        points.InsertNextPoint(end_c_pos['X'], -end_c_pos['Y'], 0)

                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0,0)
                        line.GetPointIds().SetId(1,1)
                        lines.InsertNextCell(line)
                    
                    for i in range(1,3):
                        start_key = f'start_pos{i}'
                        end_key = f'end_pos{i}'
                        if start_key in xyvalue and end_key in xyvalue:
                            start_pos = xyvalue[start_key]
                            end_pos = xyvalue[end_key]
                            points.InsertNextPoint(start_pos['X'], -start_pos['Y'], 0)
                            points.InsertNextPoint(end_pos['X'], -end_pos['Y'], 0)

                            line = vtk.vtkLine()
                            line.GetPointIds().SetId(0, points.GetNumberOfPoints() - 2)
                            line.GetPointIds().SetId(1, points.GetNumberOfPoints() - 1)
                            lines.InsertNextCell(line) 
                elif objectname == "LabelLine":    
                    print(f"Creating LabelLine for {base_filename}")
                    start_pos = xyvalue.get('start_pos', None)
                    end_pos = xyvalue.get('end_pos', None)
                    if start_pos and end_pos:
                        points.InsertNextPoint(start_pos['X'], -start_pos['Y'], 0)
                        points.InsertNextPoint(end_pos['X'], -end_pos['Y'], 0)

                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0,0)
                        line.GetPointIds().SetId(1,1)
                        lines.InsertNextCell(line)
                elif objectname == "LabelPolyLine":
                    print(f"Creating LabelPolyLine for {base_filename}")
                    pos_list = xyvalue.get('pos_list', None)
                    if pos_list:
                        for idx, pos in enumerate(pos_list):
                            points.InsertNextPoint(pos['X'], -pos['Y'], 0)
                            if idx > 0:
                                line = vtk.vtkLine()
                                line.GetPointIds().SetId(0, idx-1)
                                line.GetPointIds().SetId(1, idx)
                                lines.InsertNextCell(line)    

            if points.GetNumberOfPoints() > 0:
                poly_data.SetPoints(points)
            else:
                print(f"No points found for {base_filename}") 

            if lines.GetNumberOfCells() > 0:
                poly_data.SetLines(lines)
            else:
                print(f"No lines found for {base_filename}")    

            if points.GetNumberOfPoints() > 0 and lines.GetNumberOfCells() > 0:
                save_vtp(poly_data, output_path)
            else:
                print(f"Skipping VTP file creation for {base_filename} due to lack of data")  
    except Exception as e:
        print(f"Error processing JSON file {json_path}: {e}")

def process_all_json_files(json_dirs, output_dir):
    for json_dir in json_dirs:
        for root, dirs, files in os.walk(json_dir):
            for file_name in files:
                if file_name in files:
                    if file_name.endswith(".json"):
                        json_path = os.path.join(root, file_name)
                        create_vtp_from_json(json_path, output_dir)

json_dirs = [
    "/home/ubuntu/analysis/xray_labels"
]

output_dir = "/home/ubuntu/analysis/vtp_original" 
os.makedirs(output_dir, exist_ok=True)

process_all_json_files(json_dirs, output_dir)