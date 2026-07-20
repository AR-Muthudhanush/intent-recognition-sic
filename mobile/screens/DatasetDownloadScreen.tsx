import React, { useState } from 'react';
import { View, ScrollView, StyleSheet } from 'react-native';
import { Card, Title, Paragraph, Button, ProgressBar, Snackbar } from 'react-native-paper';

const DatasetDownloadScreen = () => {
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showSnackbar, setShowSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const handleDownload = async () => {
    setDownloading(true);
    setProgress(0);

    try {
      // Simulate download with progress
      const totalSteps = 100;
      for (let i = 0; i <= totalSteps; i++) {
        await new Promise(resolve => setTimeout(resolve, 50));
        setProgress(i / totalSteps);
      }

      setDownloaded(true);
      setSnackbarMessage('Dataset downloaded successfully');
      setShowSnackbar(true);
    } catch (error) {
      setSnackbarMessage('Failed to download dataset');
      setShowSnackbar(true);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title>Dataset Information</Title>
          <Paragraph>
            Total Samples: 50,000{'\n'}
            Size: ~8 MB{'\n'}
            Format: CSV{'\n'}
            Languages: English, Korean{'\n'}
            Coverage: 62 intents, 40 targets, 40+ spatial relations
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>CSV Columns</Title>
          <Paragraph style={styles.columnDescription}>
            command{'\n'}
            Describes the UI action in natural language.
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            intent{'\n'}
            The classified action (click, scroll, delete, etc.).
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            target{'\n'}
            The UI element type (button, input, checkbox, etc.).
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            attribute{'\n'}
            Visual properties (color, size, state, etc.).
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            label{'\n'}
            Text label displayed on the UI element.
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            index{'\n'}
            Position indicator (first, second, last, etc.).
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            relation{'\n'}
            Spatial relationship (above, below, beside, etc.).
          </Paragraph>
          <Paragraph style={styles.columnDescription}>
            reference{'\n'}
            Reference element for relative positioning.
          </Paragraph>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Download Dataset</Title>
          {downloading && (
            <>
              <Paragraph>Downloading... {Math.round(progress * 100)}%</Paragraph>
              <ProgressBar progress={progress} style={styles.progressBar} />
            </>
          )}
          {downloaded && !downloading && (
            <Paragraph style={styles.successText}>✓ Dataset downloaded</Paragraph>
          )}
        </Card.Content>
        <Card.Actions>
          <Button
            mode="contained"
            onPress={handleDownload}
            loading={downloading}
            disabled={downloading}
            style={styles.downloadButton}
          >
            {downloaded && !downloading ? 'Downloaded' : 'Download'}
          </Button>
        </Card.Actions>
      </Card>

      <Card style={styles.card}>
        <Card.Content>
          <Title>Usage</Title>
          <Paragraph>
            The dataset can be used for:
            {'\n\n'}
            • Training custom models
            {'\n'}
            • Fine-tuning the intent recognition system
            {'\n'}
            • Data analysis and visualization
            {'\n'}
            • Creating benchmarks
            {'\n'}
            • Research and academic purposes
            {'\n\n'}
            Format: CSV (comma-separated values)
            {'\n'}
            Encoding: UTF-8
            {'\n'}
            Header Row: Yes
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
  columnDescription: {
    fontSize: 12,
    marginBottom: 12,
    paddingLeft: 8,
  },
  progressBar: {
    marginTop: 12,
    height: 8,
  },
  downloadButton: {
    marginTop: 12,
  },
  successText: {
    color: '#4caf50',
    fontWeight: 'bold',
    marginTop: 12,
  },
});

export default DatasetDownloadScreen;
