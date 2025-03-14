Lensless Multi-Task
As a new paradigm in computational imaging, lensless imaging holds promise for camera miniaturization. However, lensless image reconstruction or segmentation for lensless imaging is challenging due to sensor noise, diffraction effects, and imperfect coding. To tackle the issue, we propose a customized multi-task learning framework called RecSegNet, integrating image reconstruction and segmentation into a single network to boost each other by the complementary information between the two tasks. Our RecSegNet is a Y-shaped architecture consisting of an encoder and two decoders. The encoder includes an optical-aware estimator (OE), a pyramid vision Transformer (PVT), and customized tokenized multi-layer perceptions (TMLP) to obtain long-range semantics. The two decoders, reconstruction decoder (RecD) and segmentation decoder (SegD), share the same structure for predicting the underlying scenes and segmentation maps, respectively. Furthermore, we propose the hierarchical feature mutual learning (HFML) module to drive each task by enhancing the interaction of two tasks. Extensive experiments demonstrate that the RecSegNet can accurately reconstruct the underlying scene while segmenting objects of interest from lensless imaging measurements.


PVT backbone and its pretrained model in baidu drive at \url{https://pan.baidu.com/s/1yuqEm_TsjCXCEIyF-V1mFA?pwd=a5n7} with code: a5n7 and google drive at \url{https://drive.google.com/drive/folders/1g_CSJvMQPTJdIXtDaIBfkhB3Ueai4gDT?usp=drive_link}
