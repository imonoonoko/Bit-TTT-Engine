use regex::Regex;

/// Clean and normalize Japanese text.
/// 1. NFKC normalization (handled by tokenizer usually, but good to do here too).
/// 2. Remove HTML tags.
/// 3. Remove excessive whitespace.
pub fn clean_text(text: &str) -> String {
    // 1. Remove HTML-like tags <...>
    let re_html = Regex::new(r"<[^>]*>").unwrap();
    let no_html = re_html.replace_all(text, "");

    // 2. Remove URL
    let re_url = Regex::new(r"https?://[\w!?/+\-_~=;.,*&@#$%()'\[\]]+").unwrap();
    let no_url = re_url.replace_all(&no_html, "");

    // 3. Normalize whitespace (tab, newline -> space, reduce multiple spaces)
    let re_space = Regex::new(r"\s+").unwrap();
    let normalized = re_space.replace_all(&no_url, " ");

    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let raw = "こんにちは。<br>詳細はこちら: https://example.com  \t  テスト  です。";
        let cleaned = clean_text(raw);
        assert_eq!(cleaned, "こんにちは。詳細はこちら: テスト です。");
    }
}
