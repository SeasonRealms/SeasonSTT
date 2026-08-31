# SeasonSTT

SeasonSTT is a lightweight .NET speech-to-text library for Whisper ONNX model bundles.

The current public API is intentionally small:

- path-based model loading
- byte-array WAV input
- plain string transcription output
- CPU-first execution for simple integration
- Repository: [SeasonRealms/SeasonSTT](https://github.com/SeasonRealms/SeasonSTT)
- Models: https://huggingface.co/SeasonEngine/Whisper

## Install

```bash
dotnet add package SeasonSTT
```

## Quick Start

```csharp
using SeasonSTT;

var whisper = @"../../../../../../Models/whisper-large-v3-turbo";
var voices = File.ReadAllBytes("sample.wav");

var result = Whisper.Detect(whisper, voices, "en");

Console.WriteLine(result);
```

Equivalent fully-qualified call:

```csharp
var whisper = @"../../../../../../Models/whisper-large-v3-turbo";
var voices = File.ReadAllBytes("sample.wav");
var result = SeasonSTT.Whisper.Detect(whisper, voices, "en");
```

## Input Requirements

- `voice` must be a WAV file encoded as PCM16
- mono and multi-channel WAV files are accepted
- audio is resampled internally to the model sample rate
- `defaultLanguage` should be a Whisper language code such as `en`, `zh`, or `ja`

## Model Layout

`Whisper.Detect(...)` expects the `model` directory to contain `season-whisper.json` and the ONNX files referenced by that config, typically:

```text
whisper-large-v3-turbo/
  season-whisper.json
  encoder_model.onnx
  decoder_model.onnx
  preprocess_model.onnx
```

The exact filenames come from `season-whisper.json`.

## API

Current entry point:

```csharp
public static string Detect(string model, byte[] voice, string defaultLanguage)
```

Parameter notes:

- `model`: path to the Whisper model bundle directory
- `voice`: WAV PCM16 audio bytes
- `defaultLanguage`: default language token used for decoding when the config does not force a language

Return value:

- recognized text as a single `string`

## Runtime Notes

- The current implementation forces the ONNX provider to `cpu`
- Provider fallback hooks are present in the source for future expansion
- The package depends on `Microsoft.ML.OnnxRuntime.Managed`

## Build And Pack

Build the library:

```bash
dotnet build SeasonSTT/SeasonSTT.csproj -c Release
```

Create a NuGet package:

```bash
dotnet pack SeasonSTT/SeasonSTT.csproj -c Release
```

## Repository

- GitHub: [SeasonRealms/SeasonSTT](https://github.com/SeasonRealms/SeasonSTT)

## License

SeasonSTT is distributed under the MIT License. See [LICENSE](LICENSE).
