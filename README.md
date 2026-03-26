# Complete GitHub Repository Structure (Final English Version)
## Clean, professional, fully compliant with academic open-source standards
```
├── ABIDE/                        # ABIDE dataset: standard 10-fold cross-validation (individual-level)
│   ├── run.log                   # Full training log
│   ├── test.log                  # Full testing log
│   ├── ABIDE_Demo.ipynb          # Jupyter demo for result reproduction
│   ├── dataload.py               # Dataset loading and data splitting
│   ├── main.py                   # Training script
│   └── model.py                  # Model architecture

├── ABIDE_LeaveGroupOut/          # ABIDE dataset: leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # Dataset loading and leave-group-out splitting
│   ├── main.py
│   └── model.py

├── ABIDE_SiteLeaveGroupOut/      # ABIDE dataset: site-specific leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── ABIDE_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # Dataset loading and site-level leave-group-out splitting
│   ├── main.py
│   └── model.py

├── MDD/                          # MDD dataset: standard 10-fold cross-validation (individual-level)
│   ├── run.log
│   ├── test.log
│   ├── MDD_Demo.ipynb
│   ├── dataload.py               # Dataset loading and data splitting
│   ├── main.py
│   └── model.py

├── MDD_LeaveGroupOut/            # MDD dataset: leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── MDD_LeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # Dataset loading and leave-group-out splitting
│   ├── main.py
│   └── model.py

├── MDD_SiteLeaveGroupOut/        # MDD dataset: site-specific leave-group-out cross-validation
│   ├── run.log
│   ├── test.log
│   ├── MDD_SiteLeaveGroupOut_Demo.ipynb
│   ├── dataload.py               # Dataset loading and site-level leave-group-out splitting
│   ├── main.py
│   └── model.py

├── model/                        # Core model components
│   ├── brainmsgpassing.py
│   ├── base_model.py
│   └── metrics.py

└── README.md                     # Repository documentation
```

---

# Model Checkpoints & Important Notes (English)
## 📦 Model Checkpoints (Baidu Netdisk)
Due to large file sizes, **all pre-trained model checkpoints (`ckpt_demo/`) are NOT stored in GitHub**.  
All checkpoints are available via Baidu Netdisk:

**Download Link**: MF-HGNN_Model_Checkpoints  
https://pan.baidu.com/s/1qII63kUEUc2tQDKOi-JwnQ  
**Extract Code**: t7t9

After downloading, place the `ckpt_demo/` folder into the **corresponding experimental directory** to run the Jupyter notebooks successfully.

## 📌 Key File Descriptions
- **`dataload.py`** (in each experiment folder): Responsible for **dataset loading and train/test data splitting**, including 10-fold cross-validation, leave-group-out cross-validation, and site-specific leave-group-out cross-validation.
- **`run.log` / `test.log`**: Complete training and testing records with loss and evaluation metrics.
- **`*_Demo.ipynb`**: One-click Jupyter demos for full experimental reproducibility.

---

### This version is 100% ready for GitHub
- No Chinese text left (fully English)
- Clean, academic formatting
- Matches exactly what you need for the reviewer response
- Directly copy-paste into your `README.md`

Want me to give you the **full, finalized README.md file** (ready to upload directly)?
