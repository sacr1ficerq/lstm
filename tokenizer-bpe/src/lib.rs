use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use std::fs;

#[pyclass]
pub struct BPETokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    merges: Vec<(u32, u32)>,
    merge_lookup: HashMap<(u32, u32), u32>,  
    unk_token: String,
    pad_token: String,
    unk_id: u32,
    pad_id: u32,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn new(unk_token: Option<String>, pad_token: Option<String>) -> Self {
        let unk = unk_token.unwrap_or_else(|| "<UNK>".to_string());
        let pad = pad_token.unwrap_or_else(|| "<PAD>".to_string());
        
        let mut vocab = HashMap::new();
        vocab.insert(pad.clone(), 0);
        vocab.insert(unk.clone(), 1);
        
        let mut reverse_vocab = HashMap::new();
        reverse_vocab.insert(0, pad.clone());
        reverse_vocab.insert(1, unk.clone());
        
        Self {
            vocab,
            reverse_vocab,
            merges: Vec::new(),
            merge_lookup: HashMap::new(),
            unk_token: unk,
            pad_token: pad,
            unk_id: 1,
            pad_id: 0,
        }
    }

    fn build_vocab(&mut self, texts: Vec<String>, vocab_size: usize) {
        
        let mut char_to_id = HashMap::new();
        let mut current_vocab_size = 2u32;
        
        
        let mut unique_chars = HashSet::new();
        for text in &texts {
            unique_chars.extend(text.chars());
        }
        
        let mut sorted_chars: Vec<_> = unique_chars.into_iter().collect();
        sorted_chars.sort_unstable();
        
        for ch in sorted_chars {
            let ch_str = ch.to_string();
            self.vocab.insert(ch_str.clone(), current_vocab_size);
            self.reverse_vocab.insert(current_vocab_size, ch_str);
            char_to_id.insert(ch, current_vocab_size);
            current_vocab_size += 1;
        }
        
        
        let mut sequences: Vec<Vec<u32>> = texts
            .into_iter()
            .filter_map(|text| {
                if text.is_empty() {
                    None
                } else {
                    Some(text.chars().map(|c| char_to_id[&c]).collect())
                }
            })
            .collect();
        
        let max_merges = vocab_size.saturating_sub(current_vocab_size as usize);
        if max_merges == 0 || sequences.is_empty() {
            return;
        }
        
        
        self.merge_lookup.reserve(max_merges);
        
        for _ in 0..max_merges {
            
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            
            for seq in &sequences {
                for window in seq.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }
            
            if pair_counts.is_empty() {
                break;
            }
            
            
            let (&best_pair, _) = pair_counts.iter().max_by_key(|(_, &count)| count).unwrap();
            let (first_id, second_id) = best_pair;
            
            
            let new_token = format!("{}{}", 
                &self.reverse_vocab[&first_id], 
                &self.reverse_vocab[&second_id]
            );
            
            let new_id = current_vocab_size;
            self.vocab.insert(new_token.clone(), new_id);
            self.reverse_vocab.insert(new_id, new_token);
            self.merges.push((first_id, second_id));
            self.merge_lookup.insert((first_id, second_id), new_id);
            current_vocab_size += 1;
            
            
            for seq in &mut sequences {
                if seq.len() < 2 {
                    continue;
                }
                
                let mut write_idx = 0;
                let mut read_idx = 0;
                
                while read_idx < seq.len() {
                    if read_idx + 1 < seq.len() 
                        && seq[read_idx] == first_id 
                        && seq[read_idx + 1] == second_id 
                    {
                        seq[write_idx] = new_id;
                        read_idx += 2;
                    } else {
                        seq[write_idx] = seq[read_idx];
                        read_idx += 1;
                    }
                    write_idx += 1;
                }
                
                seq.truncate(write_idx);
            }
        }
    }
    
    fn encode(&self, text: String) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }
        
        
        let mut seq: Vec<u32> = text
            .chars()
            .map(|c| {
                let s = c.to_string();
                *self.vocab.get(&s).unwrap_or(&self.unk_id)
            })
            .collect();
        
        
        for &(first_id, second_id) in &self.merges {
            if seq.len() < 2 {
                break;
            }
            
            let new_id = self.merge_lookup[&(first_id, second_id)];
            let mut write_idx = 0;
            let mut read_idx = 0;
            
            while read_idx < seq.len() {
                if read_idx + 1 < seq.len() 
                    && seq[read_idx] == first_id 
                    && seq[read_idx + 1] == second_id 
                {
                    seq[write_idx] = new_id;
                    read_idx += 2;
                } else {
                    seq[write_idx] = seq[read_idx];
                    read_idx += 1;
                }
                write_idx += 1;
            }
            
            seq.truncate(write_idx);
        }
        
        seq.into_iter().map(|id| id as usize).collect()
    }
    
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<usize>> {
        texts.into_iter().map(|t| self.encode(t)).collect()
    }
    
    fn decode(&self, token_ids: Vec<usize>) -> String {
        let mut result = String::new();
        for &id in &token_ids {
            if let Some(token) = self.reverse_vocab.get(&(id as u32)) {
                result.push_str(token);
            }
        }
        result
    }
    
    fn decode_batch(&self, token_ids_batch: Vec<Vec<usize>>) -> Vec<String> {
        token_ids_batch.into_iter().map(|ids| self.decode(ids)).collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    fn token_to_id(&self, token: String) -> Option<usize> {
        self.vocab.get(&token).map(|&id| id as usize)
    }
    
    fn id_to_token(&self, id: usize) -> Option<String> {
        self.reverse_vocab.get(&(id as u32)).cloned()
    }
    
    fn get_unk_id(&self) -> usize {
        self.unk_id as usize
    }
    
    fn get_pad_id(&self) -> usize {
        self.pad_id as usize
    }
    
    fn save(&self, path: String) -> PyResult<()> {
        let merge_strings: Vec<(String, String)> = self.merges
            .iter()
            .map(|&(a, b)| (self.reverse_vocab[&a].clone(), self.reverse_vocab[&b].clone()))
            .collect();
        
        let data = serde_json::json!({
            "vocab": self.vocab,
            "merges": merge_strings,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        });
        
        fs::write(path, serde_json::to_string_pretty(&data).unwrap())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }
    
    fn load(&mut self, path: String) -> PyResult<()> {
        let content = fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        
        let data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        self.vocab = serde_json::from_value(data["vocab"].clone()).unwrap();
        let merge_strings: Vec<(String, String)> = 
            serde_json::from_value(data["merges"].clone()).unwrap();
        
        self.unk_token = data["unk_token"].as_str().unwrap().to_string();
        self.pad_token = data["pad_token"].as_str().unwrap().to_string();
        
        self.reverse_vocab.clear();
        for (token, &id) in &self.vocab {
            self.reverse_vocab.insert(id, token.clone());
        }
        
        self.merges.clear();
        self.merge_lookup.clear();
        for (first_str, second_str) in merge_strings {
            let first_id = self.vocab[&first_str];
            let second_id = self.vocab[&second_str];
            let merged = format!("{}{}", first_str, second_str);
            let merged_id = self.vocab[&merged];
            
            self.merges.push((first_id, second_id));
            self.merge_lookup.insert((first_id, second_id), merged_id);
        }
        
        self.unk_id = self.vocab[&self.unk_token];
        self.pad_id = self.vocab[&self.pad_token];
        
        Ok(())
    }
}

#[pymodule]
fn bpe_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BPETokenizer>()?;
    Ok(())
}
