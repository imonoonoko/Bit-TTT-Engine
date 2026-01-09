use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Role {
    User,
    AI,
    System,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: String) -> Self {
        Self { role, content }
    }

    /// Formats the message for the prompt.
    /// Format: "Role: Content\n"
    pub fn to_prompt_line(&self) -> String {
        let role_str = match self.role {
            Role::User => "User",
            Role::AI => "AI",
            Role::System => "System",
        };
        format!("{}: {}\n", role_str, self.content)
    }
}
