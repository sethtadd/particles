#ifndef AUDIOPLAYER_HPP
#define AUDIOPLAYER_HPP

#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

#include <sndfile.h>
#include <portaudio.h>

class AudioPlayer
{
private:
    const int framesPerBuffer_{256};

    const char *filename_;
    SNDFILE *file_;
    SF_INFO sfinfo_;
    PaError err_;
    PaStream *stream_;
    float *buffer_;
    sf_count_t framesRead_;

    // Double buffering
    // bufferMutex_ ensures the readBuffer_ and writeBuffer_ are thread-safe
    std::mutex bufferMutex_;
    float *readBuffer_;
    float *writeBuffer_;

    // audioMutex_ ensured the AudioPlayer::play and AudioPlayer::stop are thread-safe
    std::mutex audioMutex_;
    // Audio loop in AudioPlayer::play will stop when stopPlaying_ is set to true
    std::atomic<bool> stopPlaying_{false};

public:
    AudioPlayer();
    ~AudioPlayer();

    void init(const char *filename);
    void play();
    void stop();

    bool isPlaying();

    int getAudioBufferSize();
    void copyAudioBufferData(float *dest, int size);
};

#endif // AUDIOPLAYER_HPP
