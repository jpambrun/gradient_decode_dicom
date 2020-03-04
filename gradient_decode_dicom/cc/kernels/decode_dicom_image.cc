/* Copyright 2019 Gradient Health Inc. All Rights Reserved.
   Author: Marcelo Lerendegui <marcelo@gradienthealth.io>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "dcmtk/config/osconfig.h"

#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/ofstd/ofstring.h"
#include "dcmtk/ofstd/ofstdinc.h"
#include "dcmtk/ofstd/oftypes.h"
#include <dcmtk/dcmdata/dcfilefo.h>
#include "dcmtk/dcmdata/dcistrmb.h"
#include "dcmtk/dcmdata/dcdict.h"

#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmdata/dcrledrg.h"  /* for DcmRLEDecoderRegistration */
#include "dcmtk/dcmjpeg/djdecode.h"  /* for dcmjpeg decoders */
#include "dcmtk/dcmjpeg/dipijpeg.h"  /* for dcmimage JPEG plugin */
#include "dcmtk/dcmjpls/djdecode.h"  /* for dcmjpls decoders */
#include "dcmtk/dcmimage/dipitiff.h" /* for dcmimage TIFF plugin */
#include "dcmtk/dcmimage/dipipng.h"  /* for dcmimage PNG plugin */
#include "dcmtk/dcmimage/diregist.h"

#include "fmjpeg2k/djencode.h"
#include "fmjpeg2k/djdecode.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include <cstdint>
#include <exception>

typedef uint64_t Uint64; // Uint64 not present in tensorflow::custom-op docker image dcmtk

using namespace tensorflow;


template <class T, class dtype>
void copy_to_tensor (void* src, Tensor *dst, unsigned long len) {
    auto output_flat = dst->template flat<dtype>();
    T* pixels = reinterpret_cast<T*>(src);
    for (unsigned long p = 0; p < len; p++)
    {
        output_flat(p) = (dtype)pixels[p];
    }
}

template <typename dtype>
class DecodeDICOMImageOp : public OpKernel
{
public:
    explicit DecodeDICOMImageOp(OpKernelConstruction *context) : OpKernel(context)
    {
        // Get the on_error
        OP_REQUIRES_OK(context, context->GetAttr("on_error", &on_error));

        // Get the on_error
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale));

        // Get the color_dim
        OP_REQUIRES_OK(context, context->GetAttr("color_dim", &color_dim));

        DcmRLEDecoderRegistration::registerCodecs(); // register RLE codecs
        DJDecoderRegistration::registerCodecs();     // register JPEG codecs
        DJLSDecoderRegistration::registerCodecs();   // register JPEG-LS codecs
        FMJPEG2KEncoderRegistration::registerCodecs();
	    FMJPEG2KDecoderRegistration::registerCodecs();
    }

    ~DecodeDICOMImageOp()
    {
        DcmRLEDecoderRegistration::cleanup(); // deregister RLE codecs
        DJDecoderRegistration::cleanup();     // deregister JPEG codecs
        DJLSDecoderRegistration::cleanup();   // deregister JPEG-LS codecs
        FMJPEG2KEncoderRegistration::cleanup();
        FMJPEG2KDecoderRegistration::cleanup();
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input file content tensor
        const Tensor &in_contents = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(in_contents.shape()),
                    errors::InvalidArgument(
                        "DecodeDICOMImage expects input content tensor to be scalar, but had shape: ",
                        in_contents.shape().DebugString()));

        const auto in_contents_scalar = in_contents.scalar<string>()();

        // Load Dicom Image
        DcmInputBufferStream dataBuf;
        dataBuf.setBuffer(in_contents_scalar.data(), in_contents_scalar.length());
        dataBuf.setEos();

        DcmFileFormat *dfile = new DcmFileFormat();
        dfile->transferInit();
        OFCondition cond = dfile->read(dataBuf);
        dfile->transferEnd();

        DicomImage *image = NULL;
        try
        {
            image = new DicomImage(dfile, EXS_Unknown, CIF_DecompressCompletePixelData);
        }
        catch (...)
        {
            image = NULL;
        }

        unsigned long frameWidth = 0;
        unsigned long frameHeight = 0;
        unsigned long dataSize = 0;
        unsigned long frameCount = 0;
        unsigned int samples_per_pixel = 0;

        if ((image == NULL) || (image->getStatus() != EIS_Normal))
        {
            if (on_error == "strict")
            {
                OP_REQUIRES(context, false,
                            errors::InvalidArgument("Error loading image"));
                return;
            }
            else if ((on_error == "skip") || (on_error == "lossy"))
            {
                Tensor *output_tensor = NULL;
                OP_REQUIRES_OK(context, context->allocate_output(0,
                                                                 {0},
                                                                 &output_tensor));
                return;
            }
        }

        // Get image information
        frameCount = image->getFrameCount(); // getNumberOfFrames(); starts at version DCMTK-3.6.1_20140617
        frameWidth = image->getWidth();
        frameHeight = image->getHeight();
        samples_per_pixel = image->isMonochrome() ? 1 : 3;

        // Create an output tensor shape
        TensorShape out_shape;
        if ((samples_per_pixel == 1) && (color_dim == false))
        {
            out_shape = TensorShape({frameCount, frameHeight, frameWidth});
        }
        else
        {
            out_shape = TensorShape({frameCount, frameHeight, frameWidth, samples_per_pixel});
        }

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0,
                                                         out_shape,
                                                         &output_tensor));


        unsigned long frame_pixel_count = frameHeight * frameWidth * samples_per_pixel;

        for (Uint64 f = 0; f < frameCount; f++)
        {
            const DiPixel* dipix = const_cast<DiPixel*>(  image->getInterData() );
            void* mpixels = (const_cast<DiPixel*>(dipix))->getDataPtr();
            auto len = frame_pixel_count * frameCount;
            switch (dipix->getRepresentation())
            {
            case EPR_Uint8:
                copy_to_tensor<unsigned char, dtype>(mpixels, output_tensor, len);
                break;
            case EPR_Sint8:
                copy_to_tensor<signed char, dtype>(mpixels, output_tensor, len);
                break;
            case EPR_Uint16:
                copy_to_tensor<unsigned short int, dtype>(mpixels, output_tensor, len);
                break;
            case EPR_Sint16:
                copy_to_tensor<short int, dtype>(mpixels, output_tensor, len);
                break;
            case EPR_Uint32:
                copy_to_tensor<unsigned int, dtype>(mpixels, output_tensor, len);
                break;
            case EPR_Sint32:
                copy_to_tensor<signed int, dtype>(mpixels, output_tensor, len);
                break;
            }
        }
        delete image;
        delete dfile;
    }

    string on_error;
    string scale;
    bool color_dim;
};

// Register the CPU kernels.
#define REGISTER_DECODE_DICOM_IMAGE_CPU(dtype)                                      \
    REGISTER_KERNEL_BUILDER(                                                        \
        Name("DecodeDICOMImage").Device(DEVICE_CPU).TypeConstraint<dtype>("dtype"), \
        DecodeDICOMImageOp<dtype>);

REGISTER_DECODE_DICOM_IMAGE_CPU(int8);
REGISTER_DECODE_DICOM_IMAGE_CPU(int16);
REGISTER_DECODE_DICOM_IMAGE_CPU(int32);
REGISTER_DECODE_DICOM_IMAGE_CPU(int64);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint8);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint16);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint32);
REGISTER_DECODE_DICOM_IMAGE_CPU(uint64);
REGISTER_DECODE_DICOM_IMAGE_CPU(float);
REGISTER_DECODE_DICOM_IMAGE_CPU(Eigen::half);
REGISTER_DECODE_DICOM_IMAGE_CPU(double);

#undef REGISTER_DECODE_DICOM_IMAGE_CPU
