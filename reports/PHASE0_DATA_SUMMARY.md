# Phase 0: Prepared data summary

Checked on November 11, 2026. This report describes the data **after splitting**. The inspection followed symbolic links and opened every prepared image and mask. No data, split assignments, or preprocessing were changed, and no training was run.

## Datasets and sources

**QaTa-COV19** contains chest X-rays and masks of COVID-19 infection areas. It was compiled by researchers at Qatar University and Tampere University. This project uses the v2 data: 9,258 image–mask pairs. The local source is `datasets/QaTa-COV19/QaTa-COV19/QaTa-COV19-v2/`; the dataset description is in `datasets/QaTa-COV19/readme.txt`. The public source is the [QaTa-COV19 dataset on Kaggle](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset).

**MosMedData+** contains chest CT slices and infection masks. The local source is `datasets/COVID19-CT/frames/` and `datasets/COVID19-CT/masks/`, with 2,729 pairs. The [official LViT repository](https://github.com/HUANGLIZI/LViT#1-data-preparation) points to [MedSeg](https://www.medseg.ai/covid-19) and the [COVID-19 CT scan lesion segmentation dataset on Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset) as sources for MosMedData+.

## How the splits were prepared

We used the official LViT split IDs, rather than creating random splits. The prepared filename sets match the supplied split spreadsheets exactly:

| Dataset | Split | ID reference | Missing IDs | Extra IDs | Duplicate spreadsheet IDs |
| --- | --- | --- | ---: | ---: | ---: |
| QaTa-COV19 | Train | `datasets/Train_ID.xlsx` | 0 | 0 | 0 |
| QaTa-COV19 | Validation | `datasets/Val_ID.xlsx` | 0 | 0 | 0 |
| QaTa-COV19 | Test | `Original v2 Test Set/Images folder (no test spreadsheet supplied)` | 0 | 0 | 0 |
| MosMedData+ | Train | `datasets/MosMed_split_files/Train_text_MosMedData.xlsx` | 0 | 0 | 0 |
| MosMedData+ | Validation | `datasets/MosMed_split_files/Val_text_MosMedData.xlsx` | 0 | 0 | 0 |
| MosMedData+ | Test | `datasets/MosMed_split_files/Test_text_MosMedData+.xlsx` | 0 | 0 | 0 |

For QaTa, the spreadsheet `Image` column contains names such as `mask_covid_1.png`. Removing only the leading `mask_` makes these IDs match the prepared filenames. The training and validation sets together cover all 7,145 images in the original v2 training folder. The 2,113 prepared test IDs match the original v2 test image folder exactly; there was no separate QaTa test spreadsheet to check.

For MosMedData+, the `Image` column matches prepared filenames directly. All three splits together cover the 2,729 source CT images exactly. Spreadsheet headers and blank cells were excluded from the ID comparison.

## Prepared directory structure

```text
datasets/
├── QaTa-Covid19/
│   ├── Train_Folder/
│   │   ├── img/
│   │   └── labelcol/
│   ├── Val_Folder/
│   │   ├── img/
│   │   └── labelcol/
│   └── Test_Folder/
│       ├── img/
│       └── labelcol/
└── MosMedDataPlus/
    ├── Train_Folder/
    │   ├── img/
    │   └── labelcol/
    ├── Val_Folder/
    │   ├── img/
    │   └── labelcol/
    └── Test_Folder/
        ├── img/
        └── labelcol/
```

`img/` holds images and `labelcol/` holds ground-truth masks. Each image and its mask have the same prepared filename. All prepared image and mask entries are symbolic links to source files.

## Actual counts and pairing checks

Counts below come from the prepared folders, not from the expected totals. “Missing mask” means an image has no mask with the same filename; “Missing image” means a mask has no matching image. Link counts include both images and masks.

| Dataset | Split | Images | Masks | Expected pairs | Missing masks | Missing images | Symbolic links | Broken links |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QaTa-COV19 | Train | 5,716 | 5,716 | 5,716 | 0 | 0 | 11,432 | 0 |
| QaTa-COV19 | Validation | 1,429 | 1,429 | 1,429 | 0 | 0 | 2,858 | 0 |
| QaTa-COV19 | Test | 2,113 | 2,113 | 2,113 | 0 | 0 | 4,226 | 0 |
| MosMedData+ | Train | 2,183 | 2,183 | 2,183 | 0 | 0 | 4,366 | 0 |
| MosMedData+ | Validation | 273 | 273 | 273 | 0 | 0 | 546 | 0 |
| MosMedData+ | Test | 273 | 273 | 273 | 0 | 0 | 546 | 0 |

All six splits match the expected counts. Totals are **9,258 QaTa pairs** and **2,729 MosMed pairs**.

- **QaTa-Covid19:** filename overlap is Train-Val: 0, Train-Test: 0, Val-Test: 0.
- **MosMedDataPlus:** filename overlap is Train-Val: 0, Train-Test: 0, Val-Test: 0.

These checks establish that filenames do not overlap between splits. They do not establish patient-level separation or rule out identical image content under different filenames.

Every image and mask was fully decoded after following its link. No decode errors or image–mask size mismatches were found.

| Dataset | Image and mask size | Mask pixel values | Decode errors | Size mismatches |
| --- | --- | --- | ---: | ---: |
| QaTa-Covid19 | (224, 224) pixels | [0, 255] | 0 | 0 |
| MosMedDataPlus | (512, 512) pixels | [0, 12, 17, 21, 255] | 0 | 0 |

MosMed masks include intermediate values `12`, `17`, and `21`, as well as `0` and `255`; they are not all strictly binary on disk. Their meaning was not established by this inspection. Check how the model loader interprets these values before training. No mask values were changed.

## Visual examples after splitting

The figures below were read directly from the prepared split folders. Each shows the image, its ground-truth mask, and a red mask overlay. The preview foreground is defined as mask values greater than zero; the overlay uses 45% red blending. White mask pixels show that preview foreground. For repeatability, each example is the first filename in alphabetical order with a nonempty mask in that split. Display conversion and overlay blending affect only these preview figures; source files were preserved.

### QaTa-COV19

**Train — `covid_1.png`**

![QaTa-Covid19 Train: image, ground-truth mask, and overlay](data_previews/qata-covid19_train.png)

**Validation — `covid_1002.png`**

![QaTa-Covid19 Validation: image, ground-truth mask, and overlay](data_previews/qata-covid19_val.png)

**Test — `sub-S08047_ses-E16280_run-1_bp-chest_vp-ap_dx.png`**

![QaTa-Covid19 Test: image, ground-truth mask, and overlay](data_previews/qata-covid19_test.png)

### MosMedData+

**Train — `Jun_coronacases_case10_100.png`**

![MosMedDataPlus Train: image, ground-truth mask, and overlay](data_previews/mosmeddataplus_train.png)

**Validation — `Jun_coronacases_case8_76.png`**

![MosMedDataPlus Validation: image, ground-truth mask, and overlay](data_previews/mosmeddataplus_val.png)

**Test — `Jun_coronacases_case8_157.png`**

![MosMedDataPlus Test: image, ground-truth mask, and overlay](data_previews/mosmeddataplus_test.png)

## Conclusion

Both prepared datasets are ready for model setup at the image–mask file level. MosMed mask-value handling needs to be checked in that setup before training. Counts and split membership agree with the supplied LViT references, all pairs are present, and all symbolic links resolve. No broken files or size mismatches were found. QaTa test membership was verified against the original v2 test folder because no QaTa test spreadsheet was supplied.

The prepared split folders contain images and masks only. LViT text-annotation files still need to be connected in the model setup: the MosMed source spreadsheets contain text, while the supplied QaTa spreadsheets contain split IDs only. This inspection did not verify text alignment or clinical mask accuracy. No training or preprocessing changes were performed.
