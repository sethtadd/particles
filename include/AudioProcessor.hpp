#include <vector>

class AudioProcessor
{
public:
    AudioProcessor(bool isStereo, float sampleRate);
    ~AudioProcessor();

    // Functions
    void updateAudioData(const std::vector<float> &rawAudioData);
    void computeFft();
    void process();

    // Getters
    const std::vector<float> &getFFTData() const { return fftData_; }
    const std::vector<float> &getProcessedData() const { return processedData_; }

    const float getLowFreq() const;
    const float getMidFreq() const;
    const float getHighFreq() const;

    // Utility functions
    std::vector<float> stereoToMono(const std::vector<float> &stereoData);

private:
    // Variables
    bool isStereo_;
    float sampleRate_;

    std::vector<float> audioData_;
    std::vector<float> fftData_;
    std::vector<float> processedData_;

    std::vector<float> prevAudioData_;
    std::vector<float> prevFftData_;
    std::vector<float> prevProcessedData_;
};
