import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider, MD3DarkTheme, MD3LightTheme } from 'react-native-paper';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import AsyncStorage from '@react-native-async-storage/async-storage';

import HomeScreen from './screens/HomeScreen';
import AnalyzeScreen from './screens/AnalyzeScreen';
import HistoryScreen from './screens/HistoryScreen';
import JSONViewerScreen from './screens/JSONViewerScreen';
import DatasetDownloadScreen from './screens/DatasetDownloadScreen';
import SettingsScreen from './screens/SettingsScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    loadThemePreference();
  }, []);

  const loadThemePreference = async () => {
    try {
      const savedTheme = await AsyncStorage.getItem('theme');
      if (savedTheme) {
        setIsDarkMode(savedTheme === 'dark');
      }
    } catch (error) {
      console.error('Failed to load theme:', error);
    }
  };

  const toggleTheme = async () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    try {
      await AsyncStorage.setItem('theme', newMode ? 'dark' : 'light');
    } catch (error) {
      console.error('Failed to save theme:', error);
    }
  };

  const theme = isDarkMode ? MD3DarkTheme : MD3LightTheme;

  return (
    <PaperProvider theme={theme}>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            headerShown: true,
            tabBarIcon: ({ color, size }) => {
              let iconName = 'home';

              if (route.name === 'Home') iconName = 'home';
              else if (route.name === 'Analyze') iconName = 'magnify';
              else if (route.name === 'History') iconName = 'history';
              else if (route.name === 'JSONViewer') iconName = 'code-json';
              else if (route.name === 'Download') iconName = 'download';
              else if (route.name === 'Settings') iconName = 'cog';

              return <MaterialCommunityIcons name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: '#6200ee',
            tabBarInactiveTintColor: 'gray',
          })}
        >
          <Tab.Screen
            name="Home"
            component={HomeScreen}
            options={{
              title: 'Intent Recognition',
            }}
          />
          <Tab.Screen
            name="Analyze"
            component={AnalyzeScreen}
            options={{
              title: 'Analyze Command',
            }}
          />
          <Tab.Screen
            name="History"
            component={HistoryScreen}
            options={{
              title: 'History',
            }}
          />
          <Tab.Screen
            name="JSONViewer"
            component={JSONViewerScreen}
            options={{
              title: 'JSON Viewer',
            }}
          />
          <Tab.Screen
            name="Download"
            component={DatasetDownloadScreen}
            options={{
              title: 'Dataset',
            }}
          />
          <Tab.Screen
            name="Settings"
            component={(props) => <SettingsScreen {...props} isDarkMode={isDarkMode} toggleTheme={toggleTheme} />}
            options={{
              title: 'Settings',
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
