import React from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, Button } from 'react-native-paper';

const HomeScreen = ({ navigation }: any) => {
  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>Instance-Level Intent Recognition</Title>
          <Paragraph style={styles.description}>
            An offline multilingual system for understanding natural language UI commands.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Features</Title>
          <Paragraph>
            • Offline inference - no internet required{'\n'}
            • Multilingual support (English, Korean){'\n'}
            • Robust to typos, OCR errors, and STT variations{'\n'}
            • Structured JSON output{'\n'}
            • Dark/Light mode support{'\n'}
            • History tracking{'\n'}
            • Dataset download capability
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Supported Intents</Title>
          <Paragraph>
            Click, Tap, Double Tap, Long Press, Scroll, Swipe, Drag, Drop, Delete, Copy, Paste,
            Highlight, Select, Open, Close, Play, Pause, Stop, Search, Zoom, Rotate, Move, Resize,
            Upload, Download, Share, Save, Refresh, Retry, Enable, Disable, Check, Uncheck, Expand,
            Collapse, Accept, Reject, Approve, Back, Next, Previous, Home, Settings, Login, Logout,
            Bookmark, Pin, Unpin, Start, Finish, Exit, Continue, Add, Remove, Hide, Show, Mute,
            Unmute, Install, Uninstall, Submit, Cancel, Type, Enter, Focus, Hover, Launch.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Quick Start</Title>
          <Paragraph>
            1. Go to "Analyze" tab{'\n'}
            2. Type or paste a UI command{'\n'}
            3. Tap "Analyze" to get intent prediction{'\n'}
            4. View detailed JSON output{'\n'}
            5. Copy or share results
          </Paragraph>
        </Card.Content>
        <Card.Actions>
          <Button onPress={() => navigation.navigate('Analyze')}>Go to Analyze</Button>
        </Card.Actions>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>About</Title>
          <Paragraph>
            Version: 1.0.0{'\n'}
            Model Size: &lt;10 MB{'\n'}
            Inference Latency: &lt;100 ms{'\n'}
            Dataset: 50,000 samples{'\n'}
            Languages: English, Korean
          </Paragraph>
        </Card.Content>
      </Card>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  card: {
    marginBottom: 16,
  },
  description: {
    marginTop: 8,
    fontSize: 14,
  },
});

export default HomeScreen;
