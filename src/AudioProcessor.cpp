#include <vector>
#include <iterator>
#include <stdexcept>
#include <cmath>

#include <fftw3.h>

#include "AudioProcessor.hpp"

AudioProcessor::AudioProcessor(bool isStereo, float sampleRate)
    : isStereo_(isStereo), sampleRate_(sampleRate)
{
}

AudioProcessor::~AudioProcessor() {}

/*
 * @brief Updates the audio processor with new raw audio data.
 *
 * @param audioData The raw audio data to process.
 */
void AudioProcessor::updateAudioData(const std::vector<float> &audioData)
{
    // Keep a copy of the last processed data
    if (audioData_.empty())
    {
        int size = isStereo_ ? audioData.size() / 2 : audioData.size();
        prevAudioData_ = std::vector<float>(size, 0.0f);
    }
    else
        prevAudioData_ = audioData_;

    // Copy the raw audio data
    if (isStereo_)
        audioData_ = stereoToMono(audioData); // Convert stereo data to mono data
    else
        audioData_ = audioData; // Copy the raw audio data
}

void AudioProcessor::computeFft()
{
    const int inSize = audioData_.size();
    const int outSize = inSize / 2 + 1;

    // Keep a copy of the last processed data
    if (fftData_.empty())
        prevFftData_ = std::vector<float>(outSize, 0.0f);
    else
        prevFftData_ = fftData_;

    // ----------- //
    // Compute FFT //
    // ----------- //

    float *in = (float *)fftwf_malloc(sizeof(float) * inSize);
    fftwf_complex *out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (outSize));

    // combine the last half of the previous audio data with the first half of the current audio data
    // float *overlappingAudioData = (float *)fftwf_malloc(sizeof(float) * inSize);
    // std::copy(prevAudioData_.end() - inSize / 2, prevAudioData_.end(), overlappingAudioData);
    // std::copy(audioData_.begin(), audioData_.begin() + inSize / 2, overlappingAudioData + inSize / 2);
    // in = overlappingAudioData;

    // Copy audio data to in and apply window function
    for (int i = 0; i < inSize; ++i)
    {
        float hanningWindow = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (inSize - 1)));
        in[i] = audioData_[i] * hanningWindow;
        // in[i] *= hanningWindow;
    }

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(inSize, in, out, FFTW_ESTIMATE);

    fftwf_execute(plan);

    // Compute magnitudes of frequency components
    fftData_.resize(outSize);
    for (int i = 0; i < outSize; ++i)
        fftData_[i] = 2.0f * sqrtf(out[i][0] * out[i][0] + out[i][1] * out[i][1]) / inSize;

    // Clean up
    fftwf_destroy_plan(plan);
    fftwf_free(out);
    fftwf_free(in);
}

/*
 * @brief Calculates the A-weighting gain for a given frequency.
 * Got this from https://github.com/audiojs/a-weighting/blob/master/a.js
 *
 * @param f The frequency to calculate the gain for.
 * @return The A-weighting gain for the given frequency.
 */
float a_weighting(float f)
{
    const float f2 = f * f;
    return 1.2588966f * 148840000.0f * f2 * f2 /
           ((f2 + 424.36f) * std::sqrt((f2 + 11599.29f) * (f2 + 544496.41f)) * (f2 + 148840000.0f));
}

void AudioProcessor::process()
{
    // Keep a copy of the last processed data
    if (processedData_.empty())
        prevProcessedData_ = std::vector<float>(fftData_.size(), 0.0f);
    else
        prevProcessedData_ = processedData_;

    // Copy the FFT data
    processedData_ = fftData_;

    // Apply A-weighting
    for (uint i = 0; i < processedData_.size(); ++i)
    {
        // Calculate the real-world frequency for this FFT bin
        float frequency = (i * sampleRate_) / processedData_.size();

        // Get the A-weighting gain for this frequency
        float aWeightingGain = a_weighting(frequency);

        // Apply the A-weighting gain to the FFT bin magnitude
        processedData_[i] *= aWeightingGain;
    }

    // Logarithmic scale
    // FIXME using log10(freq + 1) isn't "correct", but I'm using it to avoid negative values
    for (uint i = 0; i < processedData_.size(); ++i)
        processedData_[i] = 20.0f * log10f(processedData_[i] + 1.0f);

    // Spatial smoothing
    for (uint i = 0; i < processedData_.size(); ++i)
    {
        float prev = i > 0 ? processedData_[i - 1] : 0.0f;
        float curr = processedData_[i];
        float next = i < processedData_.size() - 1 ? processedData_[i + 1] : 0.0f;

        processedData_[i] = (prev + curr + next) / 3.0f;
    }
    // Second pass
    for (uint i = 0; i < processedData_.size(); ++i)
    {
        float prev = i > 0 ? processedData_[i - 1] : 0.0f;
        float curr = processedData_[i];
        float next = i < processedData_.size() - 1 ? processedData_[i + 1] : 0.0f;

        processedData_[i] = (prev + curr + next) / 3.0f;
    }

    // Temporal smoothing)
    float smoothingFactor = 0.8f;
    for (uint i = 0; i < processedData_.size(); ++i)
    {
        processedData_[i] = smoothingFactor * prevProcessedData_[i] + (1.0f - smoothingFactor) * processedData_[i];
    }
}

/* 
 * @brief Computes the average magnitude of the frequency bins in the low frequency range (20 - 250 Hz).
 *
 * @return The average magnitude of the low frequency range.
 */
const float AudioProcessor::getLowFreq() const
{
    int firstFreqBin = 20 * processedData_.size() / sampleRate_;
    int lastFreqBin = 250 * processedData_.size() / sampleRate_;

    float sum = 0.0f;
    for (int i = firstFreqBin; i < lastFreqBin; ++i)
        sum += processedData_[i];
    
    return sum / (lastFreqBin - firstFreqBin);
}

/* 
 * @brief Computes the average magnitude of the frequency bins in the mid frequency range (250 - 4000 Hz).
 *
 * @return The average magnitude of the mid frequency range.
 */
const float AudioProcessor::getMidFreq() const
{
    int firstFreqBin = 250 * processedData_.size() / sampleRate_;
    int lastFreqBin = 4000 * processedData_.size() / sampleRate_;

    float sum = 0.0f;
    for (int i = firstFreqBin; i < lastFreqBin; ++i)
        sum += processedData_[i];
    
    return sum / (lastFreqBin - firstFreqBin);
}

/* 
 * @brief Computes the average magnitude of the frequency bins in the high frequency range (4000 - 20000 Hz).
 *
 * @return The average magnitude of the high frequency range.
 */
const float AudioProcessor::getHighFreq() const
{
    int firstFreqBin = 4000 * processedData_.size() / sampleRate_;
    int lastFreqBin = 20000 * processedData_.size() / sampleRate_;

    float sum = 0.0f;
    for (int i = firstFreqBin; i < lastFreqBin; ++i)
        sum += processedData_[i];
    
    return sum / (lastFreqBin - firstFreqBin);
}

/*
 * @brief Converts interleaved stereo samples to mono samples.
 *
 * @param stereoData The interleaved stereo samples.
 * @return The mono samples.
 */
std::vector<float> AudioProcessor::stereoToMono(const std::vector<float> &stereoSamples)
{
    // We expect the stereo samples to be interleaved: L, R, L, R, ...
    if (stereoSamples.size() % 2 != 0)
    {
        throw std::runtime_error("Stereo samples are not properly interleaved.");
    }

    std::vector<float> monoSamples(stereoSamples.size() / 2);

    auto stereoIt = stereoSamples.begin();
    auto monoIt = monoSamples.begin();

    while (stereoIt != stereoSamples.end())
    {
        float left = *stereoIt++;
        float right = *stereoIt++;
        *monoIt++ = (left + right) / 2.0f;
    }

    return monoSamples;
}
