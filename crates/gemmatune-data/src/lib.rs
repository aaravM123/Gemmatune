use gemmatune_core::DatasetConfig;
use gemmatune_gemma::{apply_chat_template, ChatMessage};
use serde::Deserialize;
use sentencepiece_rust::SentencePieceProcessor;
use std::{fs, io, path::Path};

#[derive(Debug, Deserialize)]
struct JsonConversation {
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone)]
pub struct PreparedDataset {
    pub train: Vec<Vec<u32>>,
    pub validation: Vec<Vec<u32>>,
    pub redacted_fields: usize,
}

impl PreparedDataset {
    pub fn token_count(&self) -> usize {
        self.train
            .iter()
            .chain(&self.validation)
            .map(Vec::len)
            .sum()
    }
}

pub const GEMMA3_TOKENIZER_FILE: &str = "gemma3_cleaned_262144_v2.spiece.model";
const GEMMA_BOS_ID: u32 = 2;
const GEMMA_EOS_ID: u32 = 1;
const GEMMA_START_TURN_ID: u32 = 105;
const GEMMA_END_TURN_ID: u32 = 106;

pub struct GemmaTokenizer {
    processor: SentencePieceProcessor,
}

fn encode_gemma_template(
    template: &str,
    encode_text: impl Fn(&str) -> Result<Vec<u32>, String>,
) -> Result<Vec<u32>, String> {
    let controls = [
        ("<start_of_turn>", GEMMA_START_TURN_ID),
        ("<end_of_turn>", GEMMA_END_TURN_ID),
        ("<eos>", GEMMA_EOS_ID),
    ];
    let mut remaining = template;
    let mut ids = vec![GEMMA_BOS_ID];
    while !remaining.is_empty() {
        let next = controls
            .iter()
            .filter_map(|(text, id)| remaining.find(text).map(|offset| (offset, *text, *id)))
            .min_by_key(|(offset, _, _)| *offset);
        let Some((offset, control, id)) = next else {
            ids.extend(encode_text(remaining)?);
            break;
        };
        if offset != 0 {
            ids.extend(encode_text(&remaining[..offset])?);
        }
        ids.push(id);
        remaining = &remaining[offset + control.len()..];
    }
    Ok(ids)
}

impl GemmaTokenizer {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let processor = SentencePieceProcessor::open(path.as_ref()).map_err(|error| {
            io::Error::new(io::ErrorKind::InvalidData, format!("invalid SentencePiece model: {error}"))
        })?;
        if processor.piece_size() != 262_144 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Gemma 3 tokenizer must have 262144 pieces, found {}", processor.piece_size()),
            ));
        }
        Ok(Self { processor })
    }

    pub fn encode(&self, text: &str) -> io::Result<Vec<u32>> {
        encode_gemma_template(text, |text| {
            self.processor
                .encode(text)
                .map(|ids| ids.into_iter().map(|id| id as u32).collect())
                .map_err(|error| format!("SentencePiece encoding failed: {error}"))
        }).map(|ids| ids.into_iter().map(|id| id as u32).collect())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    pub fn decode(&self, ids: &[u32]) -> io::Result<String> {
        let ids = ids
            .iter()
            .copied()
            .filter(|id| !matches!(*id, GEMMA_BOS_ID | GEMMA_EOS_ID | GEMMA_START_TURN_ID | GEMMA_END_TURN_ID))
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        self.processor.decode(&ids).map_err(|error| {
            io::Error::new(io::ErrorKind::InvalidData, format!("SentencePiece decoding failed: {error}"))
        })
    }

    pub fn generation_stop_ids(&self) -> Vec<u32> {
        vec![GEMMA_EOS_ID, GEMMA_END_TURN_ID]
    }
}

fn redact(text: &str) -> (String, usize) {
    let mut replacements = 0;
    let words = text
        .split_whitespace()
        .map(|word| {
            let is_email = word.contains('@') && word.contains('.');
            let is_number = word.chars().filter(char::is_ascii_digit).count() >= 7;
            match (is_email, is_number) {
                (true, true) => {
                    replacements += 2;
                    "[REDACTED_EMAIL] [REDACTED_NUMBER]".to_owned()
                }
                (true, false) => {
                    replacements += 1;
                    "[REDACTED_EMAIL]".to_owned()
                }
                (false, true) => {
                    replacements += 1;
                    "[REDACTED_NUMBER]".to_owned()
                }
                (false, false) => word.to_owned(),
            }
        })
        .collect::<Vec<_>>();
    (words.join(" "), replacements)
}

pub fn prepare_chat_dataset(
    root: impl AsRef<Path>,
    config: &DatasetConfig,
    tokenizer: &GemmaTokenizer,
) -> io::Result<PreparedDataset> {
    let path = root.as_ref().join(&config.file);
    let source = fs::read_to_string(&path)?;
    let mut records = Vec::new();
    let mut redacted_fields = 0;
    for (index, line) in source
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let mut record: JsonConversation = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), index + 1),
            )
        })?;
        for message in &mut record.messages {
            if config.redact_pii {
                let (content, count) = redact(&message.content);
                message.content = content;
                redacted_fields += count;
            }
        }
        if record.messages.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{} has no messages", path.display(), index + 1),
            ));
        }
        records.push(tokenizer.encode(&apply_chat_template(&record.messages, false))?);
    }
    if records.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a training dataset needs at least two conversations",
        ));
    }
    let split = ((records.len() as f32) * config.train_split).floor() as usize;
    let split = split.clamp(1, records.len() - 1);
    let validation = records.split_off(split);
    Ok(PreparedDataset {
        train: records,
        validation,
        redacted_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_email_and_number_in_the_same_word() {
        let (redacted, count) = redact("contact ada1234567@example.com please");
        assert_eq!(
            redacted,
            "contact [REDACTED_EMAIL] [REDACTED_NUMBER] please"
        );
        assert_eq!(count, 2);
    }

    #[test]
    fn still_redacts_email_and_number_separately() {
        let (redacted, count) = redact("mail ada@example.com or call 555-123-4567");
        assert_eq!(
            redacted,
            "mail [REDACTED_EMAIL] or call [REDACTED_NUMBER]"
        );
        assert_eq!(count, 2);
    }

    #[test]
    fn encodes_gemma_control_tokens_as_their_real_ids() {
        let template = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n";
        let ids = encode_gemma_template(template, |_| Ok(vec![42])).unwrap();
        assert_eq!(
            ids,
            vec![
                GEMMA_BOS_ID,
                GEMMA_START_TURN_ID,
                42,
                GEMMA_END_TURN_ID,
                42,
                GEMMA_START_TURN_ID,
                42,
            ]
        );
    }

    #[test]
    fn writing_style_smoke_data_has_ten_train_and_two_held_out_replies() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/writing-style/conversations.jsonl");
        let conversations = fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<JsonConversation>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(conversations.len(), 12);
        let split = ((conversations.len() as f32) * 0.9).floor() as usize;
        assert_eq!((split, conversations.len() - split), (10, 2));
        for conversation in conversations {
            for message in conversation.messages.iter().filter(|message| message.role == "assistant") {
                assert!(message.content.starts_with("Noted. "));
                assert!(message.content.ends_with(" Want me to tighten that?"));
            }
        }
    }
}

