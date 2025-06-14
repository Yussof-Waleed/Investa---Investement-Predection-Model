import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  CircularProgress,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Fade,
  useTheme,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import InfoIcon from '@mui/icons-material/Info';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(0),
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
  borderRadius: theme.shape.borderRadius * 2,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  height: '100%',
  backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.background.paper,
}));

const Header = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2.5),
  borderBottom: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.background.paper,
}));

const ContentArea = styled(Box)(({ theme }) => ({
  flex: 1,
  padding: theme.spacing(3),
  overflowY: 'auto',
  backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.background.paper,
  '&::-webkit-scrollbar': {
    width: '8px',
  },
  '&::-webkit-scrollbar-track': {
    backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.grey[200],
  },
  '&::-webkit-scrollbar-thumb': {
    backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[700] : theme.palette.grey[400],
    borderRadius: '4px',
  },
}));

const StyledCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  borderRadius: theme.shape.borderRadius * 1.5,
  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  overflow: 'hidden',
}));

const FormSection = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  display: 'flex',
  gap: theme.spacing(2),
  alignItems: 'center',
}));

const ResultsSection = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(3),
  gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
  [theme.breakpoints.up('md')]: {
    gridTemplateColumns: '1fr 1fr',
  },
}));

const LoaderContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(8),
  gap: theme.spacing(3),
}));

const RiskIndicator = styled(Box)(({ theme, risk }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(0.5, 1.5),
  borderRadius: theme.shape.borderRadius * 1.5,
  color: theme.palette.common.white,
  fontWeight: 'bold',
  backgroundColor: 
    risk === 'High Risk' ? theme.palette.error.main : 
    risk === 'Medium Risk' ? theme.palette.warning.main : 
    theme.palette.success.main,
}));

const SuccessGauge = styled(Box)(({ theme, percentage }) => ({
  position: 'relative',
  height: '10px',
  width: '100%',
  backgroundColor: theme.palette.grey[200],
  borderRadius: theme.shape.borderRadius,
  marginTop: theme.spacing(1),
  marginBottom: theme.spacing(2),
  overflow: 'hidden',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    height: '100%',
    width: `${Math.max(0, Math.min(100, percentage))}%`,
    backgroundColor: 
      percentage < 30 ? theme.palette.error.main : 
      percentage < 70 ? theme.palette.warning.main : 
      theme.palette.success.main,
    transition: 'width 1s ease-in-out',
  },
}));

// Fancy loader animation component
const FancyLoader = () => {
  const theme = useTheme();
  
  return (
    <LoaderContainer>
      <Box position="relative">
        <CircularProgress 
          size={100} 
          thickness={4} 
          sx={{ color: theme.palette.primary.main }} 
        />
        <CircularProgress 
          size={100} 
          thickness={4} 
          sx={{ 
            position: 'absolute', 
            left: 0, 
            color: theme.palette.secondary.main,
            opacity: 0.7,
            animation: 'spin 2s linear infinite',
            '@keyframes spin': {
              '0%': {
                transform: 'rotate(0deg)',
              },
              '100%': {
                transform: 'rotate(360deg)',
              },
            }
          }} 
        />
        <Box 
          position="absolute" 
          top="50%" 
          left="50%" 
          sx={{ 
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <AssessmentIcon sx={{ fontSize: 40 }} />
        </Box>
      </Box>
      <Typography variant="h6">Analyzing investment potential...</Typography>
      <Typography variant="body2" color="text.secondary">
        This process may take up to 2 minutes as our AI models evaluate the project.
      </Typography>
    </LoaderContainer>
  );
};

const InvestmentAnalysis = () => {
  const theme = useTheme();
  const { projectId } = useParams(); // Extract projectId from URL
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  // Automatically fetch data when component mounts or projectId changes
  useEffect(() => {
    if (projectId) {
      handleAnalyzeProject();
    }
  }, [projectId]);

  const handleAnalyzeProject = async () => {
    if (!projectId) {
      setError("No project ID provided in the URL");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    setAnalysisResult(null);
    
    try {
      // Fetch investment prediction from API with explicit headers
      const response = await fetch(`/api/predict_investment/${projectId}/`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });
      
      // Check if the response is JSON before parsing
      const contentType = response.headers.get('content-type');
      
      if (!response.ok) {
        // Try to get error details if available in JSON format
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          throw new Error(errorData.error || `Error ${response.status}: ${response.statusText}`);
        } else {
          // Handle non-JSON error response
          const textResponse = await response.text();
          console.error('Non-JSON error response:', textResponse.substring(0, 100) + '...');
          throw new Error(`Server returned non-JSON response (${response.status})`);
        }
      }
      
      // Verify we have JSON before parsing
      if (!contentType || !contentType.includes('application/json')) {
        console.error('Unexpected content type:', contentType);
        throw new Error('Server did not return JSON. Please check your API endpoint configuration.');
      }
      
      const data = await response.json();
      setAnalysisResult(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze project');
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Render insights list with icons based on sentiment
  const renderInsights = (insights) => {
    if (!insights || insights.length === 0) return null;
    
    return (
      <List disablePadding>
        {insights.map((insight, index) => {
          // Determine icon based on sentiment analysis of the insight text
          let icon = <InfoIcon color="primary" />;
          
          if (insight.toLowerCase().includes('risk') || 
              insight.toLowerCase().includes('concern') || 
              insight.toLowerCase().includes('loss')) {
            icon = <ErrorOutlineIcon color="error" />;
          } else if (insight.toLowerCase().includes('strong') || 
                     insight.toLowerCase().includes('excellent') || 
                     insight.toLowerCase().includes('healthy')) {
            icon = <CheckCircleIcon color="success" />;
          }
          
          return (
            <ListItem key={index} alignItems="flex-start">
              <ListItemIcon sx={{ minWidth: 36 }}>
                {icon}
              </ListItemIcon>
              <ListItemText primary={insight} />
            </ListItem>
          );
        })}
      </List>
    );
  };

  return (
    <Container 
      maxWidth="xl" 
      disableGutters
      sx={{ 
        height: '100vh',
        display: 'flex',
        p: 2.5,
        backgroundColor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100],
      }}
    >
      <StyledPaper elevation={3} sx={{ flex: 1 }}>
        <Header>
          <Box display="flex" alignItems="center" gap={1.5}>
            <TrendingUpIcon sx={{ fontSize: 28 }} />
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              Investment Analysis {projectId ? `- Project ${projectId}` : ''}
            </Typography>
          </Box>
        </Header>
        
        <ContentArea>
          {!projectId && (
            <StyledCard>
              <CardContent>
                <Box display="flex" alignItems="center" gap={1.5} mb={1}>
                  <ErrorOutlineIcon color="warning" />
                  <Typography variant="h6" color="text.primary">
                    No Project Selected
                  </Typography>
                </Box>
                <Typography variant="body1">
                  This page should be accessed via a project link. Please go back and select a project.
                </Typography>
              </CardContent>
            </StyledCard>
          )}

          {isLoading && (
            <Fade in={isLoading} timeout={300}>
              <Box>
                <FancyLoader />
              </Box>
            </Fade>
          )}

          {error && (
            <Fade in={!!error} timeout={300}>
              <StyledCard sx={{ borderLeft: '4px solid', borderColor: 'error.main' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" gap={1.5} mb={1}>
                    <CancelIcon color="error" />
                    <Typography variant="h6" color="error.main">
                      Error
                    </Typography>
                  </Box>
                  <Typography variant="body1">{error}</Typography>
                </CardContent>
              </StyledCard>
            </Fade>
          )}

          {analysisResult && (
            <Fade in={!!analysisResult} timeout={500}>
              <Box>
                <Typography variant="h6" gutterBottom>
                  Analysis Results: {analysisResult.project_name}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" mb={3}>
                  Analysis performed on {new Date(analysisResult.analysis_timestamp).toLocaleString()}
                </Typography>

                <ResultsSection>
                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Success Probability
                      </Typography>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                        <Typography variant="h3" color="primary" fontWeight="bold">
                          {Math.round(analysisResult.prediction.success_percentage)}%
                        </Typography>
                        <RiskIndicator risk={analysisResult.prediction.risk_level}>
                          {analysisResult.prediction.risk_indicator} {analysisResult.prediction.risk_level}
                        </RiskIndicator>
                      </Box>
                      <SuccessGauge percentage={analysisResult.prediction.success_percentage} />
                      <Typography variant="body1" fontWeight="medium">
                        Recommendation: {analysisResult.prediction.recommendation}
                      </Typography>
                    </CardContent>
                  </StyledCard>

                  <StyledCard>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Key Insights
                      </Typography>
                      <Divider sx={{ mb: 2 }} />
                      {renderInsights(analysisResult.key_insights)}
                    </CardContent>
                  </StyledCard>
                </ResultsSection>

                <StyledCard>
                  <CardContent>
                    <Typography variant="body2" color="text.secondary">
                      Analysis powered by {analysisResult.model_info}
                    </Typography>
                  </CardContent>
                </StyledCard>
              </Box>
            </Fade>
          )}
        </ContentArea>
      </StyledPaper>
    </Container>
  );
};

export default InvestmentAnalysis;

// <Route path="/investment-analysis/:projectId" element={<InvestmentAnalysis />} />