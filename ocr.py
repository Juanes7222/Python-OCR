import torch
from PIL import Image
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
import os
import json
import re
from pathlib import Path
from datetime import datetime

class DeepSeekOCR:
    def __init__(self, model_name="deepseek-ai/deepseek-vl-7b-chat"):
        print("Cargando modelo DeepSeek-VL...")

        self.processor = VLChatProcessor.from_pretrained(model_name)

        self.model = MultiModalityCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16
        ).cuda().eval()

        self.tokenizer = self.processor.tokenizer

        print("Modelo cargado correctamente")

    def extract_text(self, image_path, prompt="Lee todo el texto escrito a mano en esta imagen y transcríbelo exactamente como aparece, línea por línea."):
        image = Image.open(image_path).convert("RGB")
        image = image.rotate(90, expand=True)
    
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        pil_images = [image]
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            outputs_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                top_p=0.95
            )

        answer = self.tokenizer.decode(
            outputs_ids[0].cpu().tolist(),
            skip_special_tokens=True
        )
        
        # Limpiar la respuesta
        if "User:" in answer:
            answer = answer.split("User:")[0].strip()
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()

        return answer

    def parsear_informacion(self, texto_raw):
        """
        Parsea el texto extraído en campos estructurados
        """
        data = {
            'nombre': None,
            'apellido': None,
            'direccion': None,
            'telefono': None,
            'fecha_nacimiento': None,
            'ciudad': None,
            'observaciones': None,
            'otros_campos': {},
            'texto_completo': texto_raw
        }
        
        # Dividir en líneas
        lineas = [l.strip() for l in texto_raw.split('\n') if l.strip()]
        
        for linea in lineas:
            linea_lower = linea.lower()
            
            # Buscar nombre
            if 'nombre' in linea_lower and not 'apellido' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['nombre'] = partes[1].strip()
            
            # Buscar apellido
            elif 'apellido' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['apellido'] = partes[1].strip()
            
            # Buscar dirección
            elif 'direcci' in linea_lower or 'domicilio' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['direccion'] = partes[1].strip()
            
            # Buscar teléfono
            elif 'tel' in linea_lower or 'celular' in linea_lower or 'móvil' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    telefono = partes[1].strip()
                    # Limpiar el teléfono de caracteres no numéricos excepto + y espacios
                    data['telefono'] = telefono
            
            # Buscar fecha de nacimiento
            elif 'fecha' in linea_lower and 'nacimiento' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['fecha_nacimiento'] = partes[1].strip()
            
            # Buscar ciudad
            elif 'ciudad' in linea_lower or 'municipio' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['ciudad'] = partes[1].strip()
            
            # Buscar observaciones
            elif 'observaci' in linea_lower or 'nota' in linea_lower:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    data['observaciones'] = partes[1].strip()
            
            # Cualquier otro campo con formato "Campo: Valor"
            elif ':' in linea:
                partes = linea.split(':', 1)
                if len(partes) == 2:
                    campo = partes[0].strip()
                    valor = partes[1].strip()
                    # Si no es un campo ya identificado, agregarlo a otros_campos
                    if campo.lower() not in ['nombre', 'apellido', 'dirección', 'teléfono', 
                                             'fecha', 'ciudad', 'observaciones']:
                        data['otros_campos'][campo] = valor
        
        return data

    def procesar_imagen_con_parseo(self, image_path):
        """
        Procesa una imagen y parsea la información
        """
        print(f"Procesando: {image_path}")
        
        # Extraer texto
        texto_raw = self.extract_text(image_path)
        
        # Parsear información
        data_parseada = self.parsear_informacion(texto_raw)
        data_parseada['archivo'] = os.path.basename(image_path)
        
        return data_parseada

    def procesar_carpeta(self, carpeta_imagenes, output_file="personas_extraidas.txt"):
        """
        Procesa todas las imágenes en una carpeta y parsea la información
        """
        carpeta = Path(carpeta_imagenes)
        extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        imagenes = [f for f in carpeta.iterdir() 
                   if f.suffix.lower() in extensiones_validas]
        
        resultados = []
        
        print(f"\nEncontradas {len(imagenes)} imágenes para procesar\n")
        
        for i, imagen_path in enumerate(sorted(imagenes), 1):
            print(f"\nProcesando {i}/{len(imagenes)}: {imagen_path.name}")
            print("-" * 60)
            
            try:
                data = self.procesar_imagen_con_parseo(str(imagen_path))
                resultados.append(data)
                
                print(f"✓ Completado")
                print(f"  Nombre: {data['nombre']}")
                print(f"  Apellido: {data['apellido']}")
                print(f"  Teléfono: {data['telefono']}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                resultados.append({
                    'archivo': imagen_path.name,
                    'error': str(e)
                })
        
        # Guardar resultados en formato legible
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INFORMACIÓN EXTRAÍDA DE IMÁGENES\n")
            f.write(f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for i, r in enumerate(resultados, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"REGISTRO #{i}\n")
                f.write(f"Archivo: {r['archivo']}\n")
                f.write(f"{'='*80}\n\n")
                
                if 'error' in r:
                    f.write(f"ERROR: {r['error']}\n")
                else:
                    f.write(f"NOMBRE:              {r['nombre'] or 'N/A'}\n")
                    f.write(f"APELLIDO:            {r['apellido'] or 'N/A'}\n")
                    f.write(f"DIRECCIÓN:           {r['direccion'] or 'N/A'}\n")
                    f.write(f"TELÉFONO:            {r['telefono'] or 'N/A'}\n")
                    f.write(f"FECHA NACIMIENTO:    {r['fecha_nacimiento'] or 'N/A'}\n")
                    f.write(f"CIUDAD:              {r['ciudad'] or 'N/A'}\n")
                    f.write(f"OBSERVACIONES:       {r['observaciones'] or 'N/A'}\n")
                    
                    if r['otros_campos']:
                        f.write(f"\nOTROS CAMPOS:\n")
                        for campo, valor in r['otros_campos'].items():
                            f.write(f"  {campo}: {valor}\n")
                    
                    f.write(f"\nTEXTO COMPLETO:\n")
                    f.write(f"{'-'*80}\n")
                    f.write(f"{r['texto_completo']}\n")
                
                f.write(f"\n")
        
        # Guardar en JSON
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        
        # Guardar en CSV
        csv_file = output_file.replace('.txt', '.csv')
        self.guardar_csv(resultados, csv_file)
        
        print(f"\n{'='*80}")
        print(f"✓ Procesamiento completado")
        print(f"{'='*80}")
        print(f"\nResultados guardados en:")
        print(f"  - {output_file} (formato legible)")
        print(f"  - {json_file} (formato JSON)")
        print(f"  - {csv_file} (formato CSV)")
        print(f"\nTotal de registros procesados: {len(resultados)}")
        
        return resultados

    def guardar_csv(self, resultados, csv_file):
        """
        Guarda los resultados en formato CSV
        """
        import csv
        
        # Obtener todos los campos posibles
        campos_base = ['archivo', 'nombre', 'apellido', 'direccion', 'telefono', 
                      'fecha_nacimiento', 'ciudad', 'observaciones']
        
        # Agregar campos adicionales encontrados
        campos_extra = set()
        for r in resultados:
            if 'otros_campos' in r:
                campos_extra.update(r['otros_campos'].keys())
        
        campos = campos_base + sorted(list(campos_extra))
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            
            for r in resultados:
                if 'error' not in r:
                    row = {campo: r.get(campo, '') for campo in campos_base}
                    
                    # Agregar campos extra
                    if 'otros_campos' in r:
                        for campo in campos_extra:
                            row[campo] = r['otros_campos'].get(campo, '')
                    
                    writer.writerow(row)


def main():
    ocr = DeepSeekOCR()
    
    # Procesar una sola imagen con parseo
    print("\n" + "="*80)
    print("PROCESANDO UNA IMAGEN CON PARSEO")
    print("="*80 + "\n")
    
    image_path = "./1.jpeg"
    
    if os.path.exists(image_path):
        data = ocr.procesar_imagen_con_parseo(image_path)
        
        print("\nInformación extraída y parseada:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    
    # Procesar múltiples imágenes de una carpeta
    print("\n" + "="*80)
    print("PROCESANDO CARPETA CON PARSEO")
    print("="*80)
    
    carpeta = "./imagenes"
    
    if os.path.exists(carpeta):
        resultados = ocr.procesar_carpeta(carpeta, output_file="personas_extraidas.txt")
    else:
        print(f"\nLa carpeta '{carpeta}' no existe.")
        print("Crea la carpeta y coloca las imágenes ahí.")


if __name__ == "__main__":
    main()