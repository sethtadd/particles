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

    // Audio loop in play() will stop when playing_ is set to false
    std::atomic<bool> playing_{false};

    // Audio loop in play() plays on its own thread
    std::thread playThread_;

public:
    AudioPlayer();
    ~AudioPlayer();

    // init() will return false if there is an error
    bool init(const char *filename);
    void startPlay();
    void stop();

    bool isPlaying();

    int getAudioBufferSize();
    void copyAudioBufferData(float *dest, int size);

private:
    void play();
};

#endif // AUDIOPLAYER_HPP
