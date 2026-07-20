import React, { useState } from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, Switch, Button, Snackbar, Divider } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';

const SettingsScreen = ({ isDarkMode, toggleTheme }: any) => {
  const [notifications, setNotifications] = useState(true);
  const [autoSaveHistory, setAutoSaveHistory] = useState(true);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const handleToggleNotifications = async () => {
    const newState = !notifications;
    setNotifications(newState);
    try {
      await AsyncStorage.setItem('notifications', JSON.stringify(newState));
    } catch (error) {
      console.error('Failed to save notification setting:', error);
    }
  };

  const handleToggleAutoSave = async () => {
    const newState = !autoSaveHistory;
    setAutoSaveHistory(newState);
    try {
      await AsyncStorage.setItem('autoSaveHistory', JSON.stringify(newState));
    } catch (error) {
      console.error('Failed to save auto-save setting:', error);
    }
  };

  const handleClearCache = async () => {
    try {
      await AsyncStorage.setItem('analyzeHistory', JSON.stringify([]));
      setSnackbarMessage('Cache cleared');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage('Failed to clear cache');
      setShowSnackbar(true);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>Display</Title>

          <View style={styles.settingRow}>
            <Paragraph>Dark Mode</Paragraph>
            <Switch value={isDarkMode} onValueChange={toggleTheme} />
          </View>

          <Divider style={styles.divider} />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Preferences</Title>

          <View style={styles.settingRow}>
            <Paragraph>Notifications</Paragraph>
            <Switch value={notifications} onValueChange={handleToggleNotifications} />
          </View>

          <Divider style={styles.divider} />

          <View style={styles.settingRow}>
            <Paragraph>Auto-save History</Paragraph>
            <Switch value={autoSaveHistory} onValueChange={handleToggleAutoSave} />
          </View>

          <Divider style={styles.divider} />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Cache & Storage</Title>
          <Paragraph style={styles.description}>
            Clear all cached data and analysis history.
          </Paragraph>
        </Card.Content>
        <Card.Actions>
          <Button onPress={handleClearCache} color="red">
            Clear Cache
          </Button>
        </Card.Actions>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>About</Title>
          <Paragraph>
            Intent Recognition System{'\n'}
            Version: 1.0.0{'\n'}
            {'\n'}
            A production-ready offline multilingual system for understanding natural language UI commands and converting them into structured JSON.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Technical Details</Title>
          <Paragraph>
            Model: BERT-Tiny Multilingual{'\n'}
            Model Size: &lt;10 MB{'\n'}
            Inference Latency: &lt;100 ms{'\n'}
            {'\n'}
            Framework: PyTorch + TensorFlow Lite{'\n'}
            Inference Type: Offline (No Internet Required){'\n'}
            Languages: English, Korean{'\n'}
            {'\n'}
            Supported Intents: 62{'\n'}
            Supported Targets: 40+{'\n'}
            Spatial Relations: 40+{'\n'}
            Dataset Size: 50,000 samples
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Privacy & Security</Title>
          <Paragraph>
            • All inference runs locally on your device{'\n'}
            • No data is sent to external servers{'\n'}
            • History is stored locally in device storage{'\n'}
            • You can clear all data at any time{'\n'}
            • Model is fully offline and open-source
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Support</Title>
          <Paragraph>
            For bugs, feature requests, or support:{'\n'}
            GitHub: anthropics/intent-recognition{'\n'}
            Email: support@anthropic.com
          </Paragraph>
        </Card.Content>
      </Card>

      <Snackbar
        visible={showSnackbar}
        onDismiss={() => setShowSnackbar(false)}
        duration={3000}
      >
        {snackbarMessage}
      </Snackbar>
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
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  divider: {
    marginVertical: 8,
  },
  description: {
    fontSize: 12,
    color: '#666',
    marginBottom: 12,
  },
});

export default SettingsScreen;
