import rawData from './siteData.json';

export interface Project {
  id: string;
  key: string;
  name: string;
  subtitle: string;
  description: string;
  detailedDescription: string;
  year: string;
  icon: string;
  portfolioImage: string;
  portfolioImageContain: boolean;
  organization: string;
  shortOrg: string;
  organizationLink: string;
  url: string;
  buttonLabel: string;
  secondaryButtonUrl: string;
  secondaryButtonLabel: string;
  isMiniPortfolio: boolean;
  categoryId: string;
  categoryName: string;
  tags: string[];
  metric: string;
  domain: string;
}

export interface Post {
  filename: string;
  slug: string;
  date: string;
  title: string;
  categories: string[];
  tags: string[];
  author: string;
  description: string;
  image: string;
  readTime: string;
  wordCount: number;
  content: string;
}

export interface Metric {
  value: string;
  label: string;
  sub: string;
}

export interface StackGroup {
  group: string;
  skills: string[];
}

export interface Profile {
  name: string;
  title: string;
  leadRole: string;
  almaMater: string;
  bio: string;
  subbio: string;
  aboutParagraphs: string[];
  picture: string;
  coordinates: string;
  location: string;
  email: string;
  personalEmail: string;
  phone: string;
  company: string;
  address: {
    entity: string;
    name: string;
    street: string;
    postalCode: string;
    city: string;
    country: string;
  };
  social: {
    github: string;
    linkedin: string;
    resume: string;
    calendly: string;
  };
  systemTelemetry: {
    status: string;
    nodeId: string;
    latency: string;
    runtime: string;
    aesthetic: string;
  };
  keyMetrics: Metric[];
  engineeringStacks: StackGroup[];
}

export interface Category {
  id: string;
  name: string;
  projectKeys: string[];
}

export interface ShopSolution {
  key: string;
  name: string;
  description: string;
  portfolioImage: string;
  portfolioImageContain: boolean;
  url: string;
  buttonLabel: string;
  secondaryButtonUrl: string;
  secondaryButtonLabel: string;
  quote?: {
    content: string;
    author: string;
  };
}

export interface SiteData {
  profile: Profile;
  projects: Project[];
  categories: Category[];
  shopSolutions: ShopSolution[];
  posts: Post[];
}

export const siteData = rawData as unknown as SiteData;
