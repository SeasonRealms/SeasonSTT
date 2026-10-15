// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonSTT

namespace Season.STT;

/// <summary>
/// Whisper decoder task, selected by the task token that follows the language
/// token in the decoder prompt.
/// </summary>
public enum WhisperTask
{
    /// <summary>
    /// Transcribe the speech in its spoken language using the
    /// <c>&lt;|transcribe|&gt;</c> task token.
    /// </summary>
    Transcribe,

    /// <summary>
    /// Translate the speech to English using the <c>&lt;|translate|&gt;</c>
    /// task token.
    /// </summary>
    Translate
}

/// <summary>
/// Loaded Whisper model sessions that can run repeated transcription without
/// reloading the ONNX files from disk.
/// </summary>
public sealed class WhisperContext : IDisposable
{
    readonly string modelDirectory;
    readonly Whisper.Config config;
    readonly InferenceSession encoderSession;
    readonly InferenceSession decoderSession;

    WhisperContext(string modelDirectory, Whisper.Config config, InferenceSession encoderSession, InferenceSession decoderSession)
    {
        this.modelDirectory = modelDirectory;
        this.config = config;
        this.encoderSession = encoderSession;
        this.decoderSession = decoderSession;
    }

    /// <summary>
    /// Loads the Whisper config and creates the encoder/decoder sessions.
    /// </summary>
    /// <param name="jsonPath">Path to `season-whisper.json`.</param>
    /// <param name="encoderPath">Path to the encoder ONNX model.</param>
    /// <param name="encoderDataPath">Path to the encoder external data; must sit next to <paramref name="encoderPath"/>.</param>
    /// <param name="decoderPath">Path to the decoder ONNX model.</param>
    /// <param name="provider">ONNX Runtime execution provider, for example <c>cpu</c>.</param>
    public static WhisperContext Load(string jsonPath, string encoderPath, string encoderDataPath, string decoderPath, string provider = "cpu")
    {
        if (string.IsNullOrWhiteSpace(jsonPath))
            throw new ArgumentException("Whisper config path is required.", nameof(jsonPath));
        if (string.IsNullOrWhiteSpace(encoderPath))
            throw new ArgumentException("Whisper encoder path is required.", nameof(encoderPath));
        if (string.IsNullOrWhiteSpace(decoderPath))
            throw new ArgumentException("Whisper decoder path is required.", nameof(decoderPath));

        // ONNX Runtime resolves external data relative to the model file, so the
        // encoder weights must sit next to the encoder graph.
        if (!string.IsNullOrWhiteSpace(encoderDataPath))
        {
            var encoderDir = Path.GetDirectoryName(Path.GetFullPath(encoderPath));
            var dataDir = Path.GetDirectoryName(Path.GetFullPath(encoderDataPath));

            if (!string.Equals(encoderDir, dataDir, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"Encoder external data '{encoderDataPath}' must sit in the same directory as '{encoderPath}'.");
            }
        }

        var config = Whisper.LoadConfig(jsonPath);

        config.Provider = string.IsNullOrWhiteSpace(provider) ? "cpu" : provider;

        var encoderSession = Whisper.CreateSession(encoderPath, config.Provider);

        try
        {
            var decoderSession = Whisper.CreateSession(decoderPath, config.Provider);

            return new WhisperContext(Path.GetDirectoryName(Path.GetFullPath(encoderPath)), config, encoderSession, decoderSession);
        }
        catch
        {
            encoderSession.Dispose();

            throw;
        }
    }

    /// <summary>
    /// Runs speech-to-text inference with the already loaded sessions.
    /// </summary>
    /// <param name="voice">WAV PCM16 audio bytes.</param>
    /// <param name="defaultLanguage">Default language code, for example <c>en</c>.</param>
    /// <param name="task">
    /// Decoder task. <see cref="WhisperTask.Transcribe"/> (default) keeps the spoken
    /// language, <see cref="WhisperTask.Translate"/> asks the decoder for an English
    /// translation and only works on models that kept the translation task (for example
    /// whisper-large-v3; the turbo fine-tune is transcription-only).
    /// </param>
    /// <returns>The decoded transcription or translation text.</returns>
    public string Detect(byte[] voice, string defaultLanguage, WhisperTask task = WhisperTask.Transcribe)
    {
        config.DefaultLanguage = string.IsNullOrWhiteSpace(defaultLanguage) ? "" : defaultLanguage;

        var audio = Whisper.ResampleToTargetRate(
            Whisper.ReadWavPcm16(voice, out int sourceSampleRate),
            sourceSampleRate,
            config.SampleRate,
            config.MaxSamples);

        var inputFeatures = Whisper.CreateInputFeatures(modelDirectory, audio, config);

        using var encoderResults = encoderSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(config.EncoderInput, inputFeatures)
        });
        var encoderHiddenStates = Whisper.CopyFloatTensor(encoderResults, config.EncoderOutput);

        string? language = string.IsNullOrWhiteSpace(config.Language)
            ? (string.IsNullOrWhiteSpace(config.DefaultLanguage) ? null : config.DefaultLanguage)
            : config.Language;

        long? languageToken = null;
        if (!string.IsNullOrWhiteSpace(language) && config.Languages.TryGetValue(language, out long configuredLanguage))
        {
            languageToken = configuredLanguage;
        }
        else if (language == null)
        {
            // Whisper prompts carry a language token right after <|sot|>; without one the
            // decoder may drift to English for non-English speech, so auto mode detects
            // the language from the encoder states instead.
            languageToken = DetectLanguage(encoderHiddenStates);
        }

        // <|translate|> sits immediately before <|transcribe|> in Whisper multilingual
        // tokenizers, so configs without the explicit token still work.
        long taskToken = task is WhisperTask.Translate
            ? (config.Translate > 0 ? config.Translate : config.Transcribe - 1)
            : config.Transcribe;

        var tokenIds = new List<long> { config.Sot };
        if (languageToken.HasValue)
        {
            tokenIds.Add(languageToken.Value);
        }

        tokenIds.Add(taskToken);
        tokenIds.Add(config.NoTime);
        int promptTokenCount = tokenIds.Count;

        for (int i = 0; i < config.MaxTokens; i++)
        {
            var decoderInputIds = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, tokenIds.Count });
            using var decoderResults = decoderSession.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(config.DecoderInput, decoderInputIds),
                NamedOnnxValue.CreateFromTensor(config.DecoderEncoderInput, encoderHiddenStates)
            });

            int nextToken = Whisper.ArgMaxLastLogit(Whisper.GetTensor(decoderResults, config.DecoderOutput).AsTensor<float>());
            if (nextToken == config.Eot)
                break;

            tokenIds.Add(nextToken);
        }

        return Whisper.DecodeTokens(tokenIds.Skip(promptTokenCount), config).Trim();
    }

    /// <summary>
    /// Picks the most likely language token from the decoder logits after
    /// <c>&lt;|sot|&gt;</c>, mirroring Whisper's own language detection pass.
    /// </summary>
    long? DetectLanguage(DenseTensor<float> encoderHiddenStates)
    {
        if (config.Languages.Count == 0)
            return null;

        var inputIds = new DenseTensor<long>(new[] { (long)config.Sot }, new[] { 1, 1 });
        using var decoderResults = decoderSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(config.DecoderInput, inputIds),
            NamedOnnxValue.CreateFromTensor(config.DecoderEncoderInput, encoderHiddenStates)
        });

        var logits = Whisper.GetTensor(decoderResults, config.DecoderOutput).AsTensor<float>();
        var values = logits.ToArray();
        int vocabSize = logits.Dimensions[^1];
        int offset = values.Length - vocabSize;

        long? bestToken = null;
        float bestValue = float.MinValue;

        foreach (long token in config.Languages.Values)
        {
            if (token < 0 || token >= vocabSize)
                continue;

            float value = values[offset + (int)token];
            if (bestToken is null || value > bestValue)
            {
                bestValue = value;
                bestToken = token;
            }
        }

        return bestToken;
    }

    public void Dispose()
    {
        decoderSession.Dispose();
        encoderSession.Dispose();
    }
}
