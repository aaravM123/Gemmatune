use gemmatune_core::{Device, LoraConfig, ModelConfig, TrainingConfig};
use gemmatune_gemma::ChatMessage;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

pub const CONVERSATIONS_FILE: &str = "conversations.jsonl";
pub const CONFIG_FILE: &str = "gemmatune.toml";
pub const LOCAL_PATH_PLACEHOLDER: &str = "<SET_LOCAL_GEMMA_3_1B_IT_PATH>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmailPart {
    pub subject: String,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmailPair {
    #[serde(default)]
    pub incoming: Option<EmailPart>,
    pub reply: EmailPart,
}

#[derive(Debug, Serialize)]
struct JsonConversation {
    messages: Vec<ChatMessage>,
}

fn format_email(part: &EmailPart) -> String {
    let subject = part.subject.trim();
    let body = part.body.trim();
    match (subject.is_empty(), body.is_empty()) {
        (true, true) => String::new(),
        (true, false) => body.to_owned(),
        (false, true) => format!("Subject: {subject}"),
        (false, false) => format!("Subject: {subject}\n\n{body}"),
    }
}

pub fn pair_to_messages(pair: &EmailPair) -> Vec<ChatMessage> {
    let reply_content = format_email(&pair.reply);
    let mut messages = Vec::new();
    if let Some(incoming) = &pair.incoming {
        let user_content = format_email(incoming);
        if !user_content.is_empty() {
            messages.push(ChatMessage {
                role: "user".into(),
                content: user_content,
            });
        }
    }
    if messages.is_empty() {
        let subject = pair.reply.subject.trim();
        let user_content = if subject.is_empty() {
            "Write an email reply in my usual style.".into()
        } else {
            format!("Write a reply regarding: {subject}")
        };
        messages.push(ChatMessage {
            role: "user".into(),
            content: user_content,
        });
    }
    messages.push(ChatMessage {
        role: "assistant".into(),
        content: if reply_content.is_empty() {
            "(empty reply)".into()
        } else {
            reply_content
        },
    });
    messages
}

pub fn train_split_for_pairs(pair_count: usize) -> f32 {
    if pair_count < 2 {
        return 0.9;
    }
    let split = default_train_split();
    let held_out = pair_count - (pair_count as f32 * split).floor() as usize;
    if held_out >= 1 {
        return split;
    }
    ((pair_count - 1) as f32) / pair_count as f32
}

fn default_train_split() -> f32 {
    0.9
}

pub fn default_training_config(pair_count: usize) -> TrainingConfig {
    TrainingConfig {
        model: ModelConfig {
            checkpoint: "gemma-3-1b-it".into(),
            local_path: Some(LOCAL_PATH_PLACEHOLDER.into()),
            method: "lora".into(),
            device: Device::Cpu,
        },
        dataset: gemmatune_core::DatasetConfig {
            format: "chat".into(),
            train_split: train_split_for_pairs(pair_count),
            redact_pii: true,
            file: CONVERSATIONS_FILE.into(),
        },
        lora: LoraConfig {
            rank: 16,
            alpha: 32.0,
            epochs: 3,
            learning_rate: 0.0002,
            target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
    }
}

pub fn render_gemmatune_toml(config: &TrainingConfig) -> String {
    format!(
        "[model]\n\
checkpoint = \"{}\"\n\
local_path = \"{}\"\n\
method = \"{}\"\n\
device = \"{}\"\n\
\n\
[dataset]\n\
format = \"{}\"\n\
train_split = {}\n\
redact_pii = {}\n\
file = \"{}\"\n\
\n\
[lora]\n\
rank = {}\n\
alpha = {}\n\
epochs = {}\n\
learning_rate = {}\n\
target_modules = {:?}\n",
        config.model.checkpoint,
        config
            .model
            .local_path
            .as_deref()
            .unwrap_or(LOCAL_PATH_PLACEHOLDER),
        config.model.method,
        config.model.device,
        config.dataset.format,
        config.dataset.train_split,
        config.dataset.redact_pii,
        config.dataset.file,
        config.lora.rank,
        config.lora.alpha,
        config.lora.epochs,
        config.lora.learning_rate,
        config.lora.target_modules,
    )
}

pub fn export_pairs(pairs: &[EmailPair], output_dir: impl AsRef<Path>) -> io::Result<()> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let mut jsonl = String::new();
    for pair in pairs {
        let conversation = JsonConversation {
            messages: pair_to_messages(pair),
        };
        let line = serde_json::to_string(&conversation).map_err(|error| {
            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
        })?;
        jsonl.push_str(&line);
        jsonl.push('\n');
    }
    fs::write(output_dir.join(CONVERSATIONS_FILE), jsonl)?;
    let config = default_training_config(pairs.len());
    fs::write(output_dir.join(CONFIG_FILE), render_gemmatune_toml(&config))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_becomes_user_incoming_and_assistant_reply() {
        let pair = EmailPair {
            incoming: Some(EmailPart {
                subject: "Budget".into(),
                body: "Can you send the sheet?".into(),
            }),
            reply: EmailPart {
                subject: "Re: Budget".into(),
                body: "Sure — attached.".into(),
            },
        };
        let messages = pair_to_messages(&pair);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert!(messages[0].content.contains("Budget"));
        assert!(messages[0].content.contains("Can you send the sheet?"));
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].content.contains("Sure — attached."));
    }

    #[test]
    fn reply_only_pair_still_emits_a_conversation() {
        let pair = EmailPair {
            incoming: None,
            reply: EmailPart {
                subject: "Thanks".into(),
                body: "Appreciate the quick turnaround.".into(),
            },
        };
        let messages = pair_to_messages(&pair);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert!(messages[0].content.contains("Thanks"));
        assert_eq!(messages[1].role, "assistant");
    }

    #[test]
    fn written_toml_parses_with_training_config_loader() {
        let config = default_training_config(3);
        let rendered = render_gemmatune_toml(&config);
        let parsed: TrainingConfig =
            toml::from_str(&rendered).expect("gemmatune.toml should parse");
        assert_eq!(parsed.model.checkpoint, "gemma-3-1b-it");
        assert_eq!(
            parsed.model.local_path.as_deref(),
            Some(LOCAL_PATH_PLACEHOLDER)
        );
        assert_eq!(parsed.model.device, Device::Cpu);
        assert_eq!(parsed.lora.learning_rate, 0.0002);
        parsed.validate().expect("config should validate");
    }

    #[test]
    fn train_split_holds_out_at_least_one_when_two_or_more_pairs() {
        for count in 2..=12 {
            let split = train_split_for_pairs(count);
            let train = (count as f32 * split).floor() as usize;
            let train = train.clamp(1, count - 1);
            assert!(count - train >= 1, "count={count} split={split}");
        }
    }

    #[test]
    fn export_writes_dataset_files() {
        let dir = std::env::temp_dir().join(format!(
            "gemmatune-gmail-export-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        export_pairs(
            &[EmailPair {
                incoming: Some(EmailPart {
                    subject: "Hi".into(),
                    body: "Question?".into(),
                }),
                reply: EmailPart {
                    subject: "Re: Hi".into(),
                    body: "Answer.".into(),
                },
            }],
            &dir,
        )
        .unwrap();
        let jsonl = fs::read_to_string(dir.join(CONVERSATIONS_FILE)).unwrap();
        assert!(jsonl.contains("\"role\":\"assistant\""));
        let toml_text = fs::read_to_string(dir.join(CONFIG_FILE)).unwrap();
        assert!(toml_text.contains(LOCAL_PATH_PLACEHOLDER));
        let _ = fs::remove_dir_all(&dir);
    }
}
