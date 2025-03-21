import h5py
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split

def extraer_patches_basicos(imagenes, etiqueta, img_size, num_canales):
    """Extrae parches básicos sin aumentación de datos"""
    patches = []
    etiquetas = []
    
    for img in imagenes:
        if img.shape[0] < img_size or img.shape[1] < img_size:
            print(f"Advertencia: Imagen demasiado pequeña {img.shape}, saltando")
            continue
            
        center_patch = img[:img_size, :img_size]
        
        if center_patch.shape == (img_size, img_size, num_canales):
            patches.append(center_patch)
            etiquetas.append(etiqueta)
        else:
            print(f"Patch con forma incorrecta: {center_patch.shape}, se esperaba {(img_size, img_size, num_canales)}")
    
    patches_array = np.array(patches)
    etiquetas_array = np.array(etiquetas)
    
    print(f"Shape de patches básicos para {'Frailejon' if etiqueta == 1 else 'NoFrailejon'}: {patches_array.shape}")
    
    return patches_array, etiquetas_array

def aplicar_data_augmentation_enhanced(X_train, y_train, img_size, num_canales, target_size=850):
    """Aplica data augmentation extendido para alcanzar un tamaño objetivo"""
    patches_aumentados = []
    etiquetas_aumentadas = []
    
    patches_aumentados.extend(X_train)
    etiquetas_aumentadas.extend(y_train)

    num_original = len(X_train)
    additional_needed = target_size - num_original
    augmentations_per_image = max(1, additional_needed // num_original)
    
    print(f"Original samples: {num_original}")
    print(f"Target: {target_size}")
    print(f"Need to generate approximately {additional_needed} new samples")
    print(f"Will apply up to {augmentations_per_image} augmentations per image")
    
    # Definir todas las posibles transformaciones
    transformations = [
        lambda img: np.fliplr(img),  # Volteado horizontal
        lambda img: np.flipud(img),  # Volteado vertical
        lambda img: np.rot90(img, k=1),  # Rotación 90°
        lambda img: np.rot90(img, k=2),  # Rotación 180°
        lambda img: np.rot90(img, k=3),  # Rotación 270°
        lambda img: np.clip(img * 1.2, 0, 255).astype(np.uint8),  # Aumento de brillo
        lambda img: (img * 0.8).astype(np.uint8),  # Disminución de brillo
        lambda img: (img * 0.7).astype(np.uint8),  # Menos brillo
        lambda img: np.clip(img + 20, 0, 255).astype(np.uint8),  # Aumento de contraste
        lambda img: np.clip(img - 20, 0, 255).astype(np.uint8),  # Disminución de contraste
        lambda img: np.clip(np.fliplr(np.rot90(img, k=1)), 0, 255).astype(np.uint8),  # Combinación 1
        lambda img: np.clip(np.flipud(np.rot90(img, k=2)), 0, 255).astype(np.uint8)  # Combinación 2
    ]
    
    for i, patch in enumerate(X_train):
        img = (patch * 255).astype(np.uint8)
        applied = 0
        
        for transform_func in transformations:
            if applied >= augmentations_per_image:
                break
                
            transformed_img = transform_func(img)
            
            if transformed_img.shape[:2] != (img_size, img_size):
                try:
                    transformed_img = transformed_img[:img_size, :img_size, :]
                except:
                    print(f"Error en el tamaño de la imagen transformada: {transformed_img.shape}")
                    continue
            
            patches_aumentados.append(transformed_img / 255.0)
            etiquetas_aumentadas.append(y_train[i])
            applied += 1
        
        if len(patches_aumentados) >= target_size:
            break
    
    patches_array = np.array(patches_aumentados[:target_size])
    etiquetas_array = np.array(etiquetas_aumentadas[:target_size])
    
    print(f"Tamaño final después de aumentación: {patches_array.shape}")
    
    return patches_array, etiquetas_array

def import_imagenes(use_enhanced_method=False, test_size=0.2, random_state=42, target_size=850):

    archivo_h5 = "data_F"
    urllib.request.urlretrieve(
        "https://github.com/sergiomora03/AdvancedTopicsAnalytics/raw/main/datasets/data_F", 
        archivo_h5
    )
    
    if use_enhanced_method:
        with h5py.File(archivo_h5, 'r') as FF:
            fraile = np.array(FF.get('Frailejon'))
            nofraile = np.array(FF.get('NoFrailejon'))
        
        print(f"Número de imágenes originales - Frailejon: {len(fraile)}, NoFrailejon: {len(nofraile)}")
        print(f"Forma de arrays originales - Frailejon: {fraile.shape}, NoFrailejon: {nofraile.shape}")
        
        img_size = 70  
        num_canales = 3  

        fraile_patches, y_fraile = extraer_patches_basicos(fraile, 1, img_size, num_canales)
        nofraile_patches, y_nofraile = extraer_patches_basicos(nofraile, 0, img_size, num_canales)
        
        X = np.vstack((fraile_patches, nofraile_patches))
        y = np.concatenate((y_fraile, y_nofraile))

        X = X.astype('float32') / 255.0
        
        print(f"Forma de X antes de split: {X.shape}")
        print(f"Forma de y antes de split: {y.shape}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_augmented, y_train_augmented = aplicar_data_augmentation_enhanced(
            X_train, y_train, img_size, num_canales, target_size=target_size
        )
        
        print(f"Forma de X_train después de aumentación: {X_train_augmented.shape}")
        print(f"Forma de y_train después de aumentación: {y_train_augmented.shape}")
        
        return X_train_augmented, X_val, y_train_augmented, y_val
    
    else:
        FF = h5py.File(archivo_h5, 'r')
        
        CTF = FF.get('Frailejon')
        fraile = np.array(CTF)

        CTNF = FF.get('NoFrailejon')
        nofraile = np.array(CTNF)

        n = fraile.shape[0]
        m = nofraile.shape[0]

        fraile2 = []
        nofraile2 = []

        r = m

        for l in range(2):
            if(l==1):
                r = n
            for i in range(0,r,1):
                for j in range(0,5,1):
                    if (j==0):
                        r1=0
                        c1=0
                        r2=70
                        c2=70
                        if(l==0):
                            x = nofraile[i,r1:r2,c1:c2,]
                            nofraile2.append(x)
                        if(l==1):
                            x = fraile[i,r1:r2,c1:c2,]
                            fraile2.append(x)
                    if (j==1):
                        r1=r1+30
                        r2=r2+30
                        if(l==0):
                            x = nofraile[i,r1:r2,c1:c2,]
                            nofraile2.append(x)
                        if(l==1):
                            x = fraile[i,r1:r2,c1:c2,]
                            fraile2.append(x)
                    if(j==2):
                        c1=c1+30
                        c2=c2+30
                        if(l==0):
                            x = nofraile[i,r1:r2,c1:c2,]
                            nofraile2.append(x)
                        if(l==1):
                            x = fraile[i,r1:r2,c1:c2,]
                            fraile2.append(x)
                    if(j==3):
                        r1=0
                        r2=70
                        if(l==0):
                            x = nofraile[i,r1:r2,c1:c2,]
                            nofraile2.append(x)
                        if(l==1):
                            x = fraile[i,r1:r2,c1:c2,]
                            fraile2.append(x)
                    if(j==4):
                        r1=15
                        c1=15
                        r2=85
                        c2=85
                        if(l==0):
                            x = nofraile[i,r1:r2,c1:c2,]
                            nofraile2.append(x)
                        if(l==1):
                            x = fraile[i,r1:r2,c1:c2,]
                            fraile2.append(x)

        nofraile2 = np.asarray(nofraile2)
        fraile2 = np.asarray(fraile2)

        CT_x2 = np.concatenate((fraile2, nofraile2))

        CT_x_columna = CT_x2.reshape(CT_x2.shape[0], -1).T

        CT_xn = CT_x_columna/255.

        CT_y = np.array([1]*fraile2.shape[0] + [0]*nofraile2.shape[0])
        
        X_train, X_test, y_train, y_test = train_test_split(
            CT_xn.T, CT_y, test_size=test_size, random_state=random_state, stratify=CT_y
        )
        

        transformations = [
            lambda img: np.fliplr(img),  # Volteado horizontal
            lambda img: np.flipud(img),  # Volteado vertical
            lambda img: np.rot90(img, k=1),  # Rotación 90°
            lambda img: np.rot90(img, k=2),  # Rotación 180°
            lambda img: np.rot90(img, k=3),  # Rotación 270°
            lambda img: np.clip(img * 1.2, 0, 255).astype(np.uint8),  # Aumento de brillo
            lambda img: (img * 0.8).astype(np.uint8),  # Disminución de brillo
            lambda img: (img * 0.7).astype(np.uint8),  # Menos brillo
            lambda img: np.clip(img + 20, 0, 255).astype(np.uint8),  # Aumento de contraste
            lambda img: np.clip(img - 20, 0, 255).astype(np.uint8),  # Disminución de contraste
            lambda img: np.clip(np.fliplr(np.rot90(img, k=1)), 0, 255).astype(np.uint8),  # Combinación 1
            lambda img: np.clip(np.flipud(np.rot90(img, k=2)), 0, 255).astype(np.uint8)  # Combinación 2
        ]
        
        img_size = int(np.sqrt(X_train.shape[1] / 3))  
        X_train_reshaped = X_train.reshape(X_train.shape[0], img_size, img_size, 3)
        X_train_reshaped = (X_train_reshaped * 255).astype(np.uint8)
        
        augmented_images = []
        augmented_labels = []

        for i, img in enumerate(X_train_reshaped):
            label = y_train[i]

            for transform in transformations:
                try:
                    transformed_img = transform(img)
                    augmented_images.append(transformed_img)
                    augmented_labels.append(label)
                except Exception as e:
                    print(f"Error al aplicar transformación: {e}")

        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        augmented_images = augmented_images / 255.
        
        augmented_images_flat = augmented_images.reshape(augmented_images.shape[0], -1)
        
        X_train_augmented = np.vstack([X_train, augmented_images_flat])
        y_train_augmented = np.concatenate([y_train, augmented_labels])
        
        X_train_augmented = X_train_augmented.T
        X_test = X_test.T
        
        y_train_augmented = y_train_augmented.reshape(-1, 1).T
        y_test = y_test.reshape(-1, 1).T
        
        print(f"Dimensión de X_train: {X_train_augmented.shape}")
        print(f"Dimensión de X_test: {X_test.shape}")
        print(f"Dimensión de y_train: {y_train_augmented.shape}")
        print(f"Dimensión de y_test: {y_test.shape}")
        
        return X_train_augmented, X_test, y_train_augmented, y_test