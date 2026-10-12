#Random Forest baseline


import os
from pathlib import Path
import sys

env_root = Path(sys.prefix)
gdal_xsd_files = list(env_root.rglob("gdalvrt.xsd"))
if gdal_xsd_files:
    os.environ['GDAL_DATA'] = str(gdal_xsd_files[0].parent)
    print(f"GDAL_DATA set to: {gdal_xsd_files[0].parent}")
else:
    print("GDAL data directory not find")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
import time
import rasterio
import warnings
warnings.filterwarnings("ignore")


DATA_ROOT = Path(
    r"D:\data_storage\geo_data\remote_sensed_data\EuroSAT\raw\EuroSATallBands\ds\images\remote_sensing\otherDatasets\sentinel_2\tif"
)

CLASSES = sorted([
    "AnnualCrop","Forest","HerbaceousVegetation","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","seaLake"
])

print("EuroSAT Multispectral Land Cover Classification")
print(f"Number Of Classes:{len(CLASSES)}")
print(f"Class List:{CLASSES}")


def load_eurosat(data_root,classes,max_per_class=None):
    x_list,y_list = [],[]
    for label_idx,cls in enumerate(classes):
        cls_dir = data_root / cls
        if not cls_dir.exists():
            print(f"Dose Not Exist:{cls_dir}")
            continue
        tif_files = sorted(cls_dir.glob("*.tif"))
        if max_per_class:
            tif_files = tif_files[:max_per_class]
        for tif_path in tif_files:
            try:
                with rasterio.open(tif_path) as src:
                    img = src.read()
                x_list.append(img)
                y_list.append(label_idx)
            except Exception as e:
                print(f"skip{tif_path.name}:{e}")
        print(f"{cls:<25}Loaded {len(tif_files)} images")
    x = np.array(x_list,dtype=np.float32)
    y = np.array(y_list,dtype=np.float32)
    return x,y

print("Loading Data")
t0 = time.time()
x,y = load_eurosat(DATA_ROOT, CLASSES, max_per_class=None)
print(f"Elapsed:{time.time() - t0}s")
print(f"x shape:{x.shape},y shape:{y.shape}")


x = x / 1000
N = x.shape[0]
x_flat = x.reshape(N,-1)
print(f"x_flat shape:{x_flat.shape}")


x_train,x_test,y_train,y_test = train_test_split(
    x_flat,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training Set:{x_train.shape[0]}")
print(f"Test Set:{x_test.shape[0]}")


print("start train")
t0 = time.time()

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

clf.fit(x_train,y_train)
train_time = time.time() - t0
print(f"Training Finished  Elapsed:{train_time:.1f}s")
print(f"OOB Score:{clf.oob_score_:.4f}")


y_pred =clf.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(f"Overall Accuracy:{acc:.4f}")
print("classification report:")
print(classification_report(y_test,y_pred,target_names=CLASSES,digits=4))


cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,annot=True,fmt="d",cmap="Blues",
    xticklabels=CLASSES,yticklabels=CLASSES,
)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title(f"EuroSAT Random Forest Confusion Matrix(OA={acc:.4f})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png",dpi=150)
plt.show()


importances = clf.feature_importances_
importances_reshaped = importances.reshape(13,64,64)
band_importance = importances_reshaped.sum(axis=(1,2))

plt.figure(figsize=(10,5))
plt.bar(range(1,14),band_importance,color="steelblue")
plt.xlabel("Sentinel-2 Band Index")
plt.ylabel("Sum of Feature Importances")
plt.title("Feature Importance of Sentinel-2 Bands")
plt.tight_layout()
plt.savefig("rf_band_importance.png",dpi=150)
plt.show()

top_bands = np.argsort(band_importance)[::-1][:5]
print("\nTop 5 Most Important Bands(0-based indexing):")
for b in top_bands:
    print(f"Band{b+1}:Importance {band_importance[b]:.4f}")

print("\nFinished")
